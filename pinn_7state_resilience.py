"""PINN solution for seven-state hydrogen refueling station resilience.

This public version contains only the calculations required for the paper:
1. training the gradient-calibrated PINN;
2. predicting the probabilities of states S0-S6; and
3. calculating resistance, absorption, recovery, and robustness.

The transition-rate parameters are aggregated values estimated from anonymized
fault records of an operating hydrogen refueling station in China.

Environment variables can be used to change the main settings, for example:
    PINN_EPOCHS=80000
    PINN_ROBUSTNESS_EPOCHS=80000
    PINN_N_R=8000
    PINN_N_STAR=3001
    PINN_T_END=150000
    PINN_OUTPUT_DIR=results
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


DTYPE = "float32"
STATE_DIM = 7
STATE_NAMES = [f"S{i}" for i in range(STATE_DIM)]
ACCEPTABLE_STATES = [0, 1, 2, 3]
FAILURE_STATES = [4, 5, 6]
SEED = 12

# Time-dependent transition rates from S0 to S1-S6:
# lambda_k(t) = LAMBDA_INTERCEPTS[k] + LAMBDA_SLOPES[k] * t.
# Units are h^-1 for the intercepts and h^-2 for the slopes.
LAMBDA_INTERCEPTS = np.array(
    [4.50e-5, 3.50e-5, 2.50e-5, 5.00e-6, 3.00e-6, 1.50e-6],
    dtype=np.float64,
)
LAMBDA_SLOPES = np.array(
    [5.00e-8, 4.00e-8, 3.00e-8, 5.00e-9, 3.00e-9, 1.50e-9],
    dtype=np.float64,
)

# Recovery rates from S1-S6 to S0, in h^-1.
RECOVERY_RATES = np.array(
    [2.50e-2, 2.20e-2, 1.80e-2, 3.50e-3, 3.00e-3, 2.50e-3],
    dtype=np.float64,
)

tf.keras.backend.set_floatx(DTYPE)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def transition_rates_np(t: np.ndarray) -> np.ndarray:
    """Return the six time-dependent transition rates at time t."""
    t = np.asarray(t, dtype=np.float64)
    return LAMBDA_INTERCEPTS + np.expand_dims(t, axis=-1) * LAMBDA_SLOPES


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration with a zero initial value."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(y, dtype=np.float64)
    result[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return result


def build_collocation_points(t_end: float, nominal_count: int) -> np.ndarray:
    """Generate residual points over the full, medium, and early time ranges."""
    n_full = nominal_count // 3
    n_mid = nominal_count // 3
    n_early = nominal_count - n_full - n_mid

    full = np.linspace(0.0, t_end, n_full, dtype=np.float32)
    medium = np.linspace(0.0, min(5000.0, t_end), n_mid, dtype=np.float32)
    early = np.linspace(0.0, min(500.0, t_end), n_early, dtype=np.float32)
    points = np.unique(np.concatenate([full, medium, early]))
    return points.reshape(-1, 1)


def make_optimizer() -> tf.keras.optimizers.Optimizer:
    """Create the Adam optimizer used in the study."""
    schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=float(os.getenv("PINN_LR_START", "3e-4")),
        decay_steps=int(os.getenv("PINN_LR_DECAY_STEPS", "30000")),
        end_learning_rate=float(os.getenv("PINN_LR_END", "2e-5")),
        power=float(os.getenv("PINN_LR_POWER", "0.5")),
    )
    return tf.keras.optimizers.Adam(learning_rate=schedule, epsilon=1e-8)


class SevenStatePINN:
    """Gradient-calibrated PINN for the seven-state Markov equations."""

    def __init__(
        self,
        initial_time: np.ndarray,
        initial_probability: np.ndarray,
        residual_points: np.ndarray,
        gate_time: float = 50.0,
        absorbing_failure: bool = False,
    ) -> None:
        initial_time = np.asarray(initial_time, dtype=np.float32).reshape(-1, 1)
        initial_probability = np.asarray(initial_probability, dtype=np.float32)
        residual_points = np.asarray(residual_points, dtype=np.float32).reshape(-1, 1)

        if initial_probability.shape[1] != STATE_DIM:
            raise ValueError("The initial probability must contain seven states.")

        self.t0 = residual_points.min(axis=0).astype(np.float32)
        self.t1 = residual_points.max(axis=0).astype(np.float32)
        self.time_scale = (self.t1 - self.t0).astype(np.float32)
        if np.any(self.time_scale == 0.0):
            raise ValueError("The residual points must span a non-zero time interval.")

        initial_time_normalized = (initial_time - self.t0) / self.time_scale
        residual_time_normalized = (residual_points - self.t0) / self.time_scale

        self.initial_time_tf = tf.convert_to_tensor(initial_time_normalized, dtype=tf.float32)
        self.initial_probability_tf = tf.convert_to_tensor(initial_probability, dtype=tf.float32)
        self.initial_state_tf = tf.convert_to_tensor(initial_probability[0:1], dtype=tf.float32)
        self.residual_time_tf = tf.convert_to_tensor(residual_time_normalized, dtype=tf.float32)
        self.residual_physical_time_tf = tf.convert_to_tensor(residual_points, dtype=tf.float32)
        self.time_scale_tf = tf.convert_to_tensor(self.time_scale.reshape(1, -1), dtype=tf.float32)
        self.recovery_rates_tf = tf.convert_to_tensor(RECOVERY_RATES, dtype=tf.float32)
        self.gate_rate = tf.constant(float(self.time_scale[0] / gate_time), dtype=tf.float32)
        self.absorbing_failure = absorbing_failure
        self.model = self._build_model()
        self.loss_history: list[dict[str, float]] = []

    @staticmethod
    def _build_model() -> tf.keras.Model:
        """Build the 3 x 96 fully connected network used in the study."""
        log_scale_slow = np.float32(np.log1p(100.0))
        log_scale_fast = np.float32(np.log1p(5000.0))

        model = tf.keras.Sequential(name="seven_state_pinn")
        model.add(tf.keras.layers.Input(shape=(1,), name="normalized_time"))
        model.add(
            tf.keras.layers.Lambda(
                lambda tau: tf.concat(
                    [
                        tau,
                        tf.math.log1p(100.0 * tau) / log_scale_slow,
                        tf.math.log1p(5000.0 * tau) / log_scale_fast,
                    ],
                    axis=1,
                ),
                name="time_features",
            )
        )
        for layer_index in range(3):
            model.add(
                tf.keras.layers.Dense(
                    96,
                    activation="tanh",
                    kernel_initializer="glorot_normal",
                    name=f"hidden_{layer_index + 1}",
                )
            )
        model.add(tf.keras.layers.Dense(STATE_DIM, name="state_logits"))
        return model

    def solution(self, normalized_time: tf.Tensor) -> tf.Tensor:
        """Return state probabilities satisfying the initial and probability constraints."""
        logits = self.model(normalized_time)
        learned_probability = tf.nn.softmax(logits, axis=1)
        gate = 1.0 - tf.exp(-self.gate_rate * normalized_time)
        return (1.0 - gate) * self.initial_state_tf + gate * learned_probability

    @staticmethod
    def transition_rates_tf(physical_time: tf.Tensor) -> tf.Tensor:
        """TensorFlow form of the six time-dependent transition rates."""
        intercepts = tf.constant(LAMBDA_INTERCEPTS.reshape(1, -1), dtype=tf.float32)
        slopes = tf.constant(LAMBDA_SLOPES.reshape(1, -1), dtype=tf.float32)
        return intercepts + physical_time * slopes

    def markov_rhs(self, probability: tf.Tensor, physical_time: tf.Tensor) -> tf.Tensor:
        """Return the right-hand side of the seven-state balance equations."""
        p0 = probability[:, 0:1]
        abnormal = probability[:, 1:]
        transition_rates = self.transition_rates_tf(physical_time)
        recovery_rates = tf.reshape(self.recovery_rates_tf, (1, -1))

        if self.absorbing_failure:
            rhs0 = -tf.reduce_sum(transition_rates, axis=1, keepdims=True) * p0
            rhs0 += tf.reduce_sum(
                recovery_rates[:, :3] * abnormal[:, :3], axis=1, keepdims=True
            )
            rhs_acceptable = (
                transition_rates[:, :3] * p0
                - recovery_rates[:, :3] * abnormal[:, :3]
            )
            rhs_failure = transition_rates[:, 3:] * p0
            return tf.concat([rhs0, rhs_acceptable, rhs_failure], axis=1)

        rhs0 = -tf.reduce_sum(transition_rates, axis=1, keepdims=True) * p0
        rhs0 += tf.reduce_sum(recovery_rates * abnormal, axis=1, keepdims=True)
        rhs_abnormal = transition_rates * p0 - recovery_rates * abnormal
        return tf.concat([rhs0, rhs_abnormal], axis=1)

    def residual_losses(self) -> list[tf.Tensor]:
        """Calculate one residual loss for each state-balance equation."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.residual_time_tf)
            probability = self.solution(self.residual_time_tf)
            components = [probability[:, i : i + 1] for i in range(STATE_DIM)]

        derivatives = tf.concat(
            [tape.gradient(component, self.residual_time_tf) for component in components],
            axis=1,
        )
        del tape

        rhs = self.markov_rhs(probability, self.residual_physical_time_tf)
        residual = derivatives - self.time_scale_tf * rhs
        return [
            tf.reduce_mean(tf.square(residual[:, i : i + 1]))
            for i in range(STATE_DIM)
        ]

    def _flatten_gradients(self, gradients: list[tf.Tensor | None]) -> tf.Tensor:
        flat_gradients = []
        for gradient, variable in zip(gradients, self.model.trainable_variables):
            if gradient is None:
                gradient = tf.zeros_like(variable)
            flat_gradients.append(tf.reshape(gradient, [-1]))
        return tf.concat(flat_gradients, axis=0)

    def _restore_gradient_shapes(self, flat_gradient: tf.Tensor) -> list[tf.Tensor]:
        gradients = []
        start = tf.constant(0, dtype=tf.int32)
        for variable in self.model.trainable_variables:
            size = tf.size(variable)
            gradients.append(tf.reshape(flat_gradient[start : start + size], tf.shape(variable)))
            start += size
        return gradients

    def calibrated_gradients(self):
        """Project conflicting gradients among the seven residual losses."""
        with tf.GradientTape(persistent=True) as tape:
            residual_terms = self.residual_losses()
            residual_loss = tf.add_n(residual_terms)

        state_gradients = [
            self._flatten_gradients(
                tape.gradient(state_loss, self.model.trainable_variables)
            )
            for state_loss in residual_terms
        ]
        del tape

        calibrated = []
        conflict_count = tf.constant(0.0, dtype=tf.float32)
        epsilon = tf.constant(1e-12, dtype=tf.float32)

        for i, gradient_i in enumerate(state_gradients):
            projected = gradient_i
            for j, gradient_j in enumerate(state_gradients):
                if i == j:
                    continue
                inner_product = tf.reduce_sum(projected * gradient_j)
                norm_squared = tf.reduce_sum(gradient_j * gradient_j)
                has_conflict = inner_product < 0.0
                projected = tf.cond(
                    has_conflict,
                    lambda: projected
                    - inner_product / (norm_squared + epsilon) * gradient_j,
                    lambda: projected,
                )
                conflict_count += tf.cast(has_conflict, tf.float32)
            calibrated.append(projected)

        combined_gradient = self._restore_gradient_shapes(tf.add_n(calibrated))
        initial_error = tf.reduce_mean(
            tf.square(
                self.initial_probability_tf - self.solution(self.initial_time_tf)
            )
        )
        return initial_error, residual_loss, residual_terms, combined_gradient, conflict_count

    def train(self, iterations: int, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """Train the PINN using gradient calibration."""

        @tf.function
        def train_step():
            outputs = self.calibrated_gradients()
            initial_error, residual_loss, residual_terms, gradients, conflicts = outputs
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return initial_error, residual_loss, residual_terms, conflicts

        for iteration in range(1, iterations + 1):
            initial_error, residual_loss, residual_terms, conflicts = train_step()
            row = {
                "iteration": iteration,
                "initial_condition_error": float(initial_error.numpy()),
                "residual_loss": float(residual_loss.numpy()),
                "conflict_count": float(conflicts.numpy()),
            }
            for state_index, value in enumerate(tf.stack(residual_terms).numpy()):
                row[f"residual_{STATE_NAMES[state_index]}"] = float(value)
            self.loss_history.append(row)

            if iteration == 1 or iteration % 1000 == 0 or iteration == iterations:
                print(
                    f"iteration {iteration:06d}: "
                    f"residual_loss={row['residual_loss']:.6e}, "
                    f"conflicts={row['conflict_count']:.0f}"
                )

    def predict(self, physical_time: np.ndarray) -> np.ndarray:
        """Predict the seven state probabilities at specified physical times."""
        physical_time = np.asarray(physical_time, dtype=np.float32).reshape(-1, 1)
        normalized_time = (physical_time - self.t0) / self.time_scale
        prediction = self.solution(
            tf.convert_to_tensor(normalized_time, dtype=tf.float32)
        ).numpy()
        prediction = np.maximum(prediction, 0.0)
        return prediction / np.sum(prediction, axis=1, keepdims=True)


def calculate_resilience_indices(
    times: np.ndarray,
    state_probabilities: np.ndarray,
    first_passage_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Calculate the resilience indices used in the paper."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    probability = np.asarray(state_probabilities, dtype=np.float64)
    first_passage_probability = np.asarray(first_passage_probabilities, dtype=np.float64)

    resistance = probability[:, 0]
    absorption = np.sum(probability[:, ACCEPTABLE_STATES], axis=1)

    total_departure_rate = np.sum(transition_rates_np(times), axis=1)
    cumulative_departure_hazard = cumulative_trapezoid(total_departure_rate, times)
    no_departure_survival = np.exp(-cumulative_departure_hazard)
    has_left_intact = 1.0 - no_departure_survival
    returned_to_intact = np.maximum(resistance - no_departure_survival, 0.0)
    recovery = np.divide(
        returned_to_intact,
        has_left_intact,
        out=np.zeros_like(returned_to_intact),
        where=has_left_intact > 1e-12,
    )
    recovery = np.clip(recovery, 0.0, 1.0)

    # In the auxiliary model, S4-S6 are absorbing. Their complement is the
    # probability that the station has not yet entered a failure state.
    robustness_survival = 1.0 - np.sum(
        first_passage_probability[:, FAILURE_STATES], axis=1
    )
    robustness_survival = np.clip(robustness_survival, 0.0, 1.0)
    robustness_time_margin = cumulative_trapezoid(robustness_survival, times)

    return pd.DataFrame(
        {
            "time_h": times,
            "resistance": resistance,
            "absorption": absorption,
            "recovery": recovery,
            "robustness_survival": robustness_survival,
            "robustness_time_margin_h": robustness_time_margin,
        }
    )


def build_summary(indices: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary of the final resilience results."""
    time = indices["time_h"].to_numpy()
    duration = time[-1] - time[0]
    resistance = indices["resistance"].to_numpy()
    absorption = indices["absorption"].to_numpy()
    recovery = indices["recovery"].to_numpy()
    peak_recovery_index = int(np.argmax(recovery))
    return pd.DataFrame(
        [
            {"indicator": "Final resistance", "value": resistance[-1]},
            {
                "indicator": "Time-averaged resistance",
                "value": np.trapz(resistance, time) / duration,
            },
            {"indicator": "Final absorption", "value": absorption[-1]},
            {
                "indicator": "Time-averaged absorption",
                "value": np.trapz(absorption, time) / duration,
            },
            {"indicator": "Final recovery", "value": recovery[-1]},
            {"indicator": "Peak recovery", "value": recovery[peak_recovery_index]},
            {"indicator": "Peak recovery time (h)", "value": time[peak_recovery_index]},
            {
                "indicator": "Final robustness time margin (h)",
                "value": indices["robustness_time_margin_h"].iloc[-1],
            },
        ]
    )


def main() -> None:
    """Train the two required PINNs and export probabilities and indices."""
    t_end = float(os.getenv("PINN_T_END", "150000"))
    nominal_residual_points = int(os.getenv("PINN_N_R", "8000"))
    prediction_points = int(os.getenv("PINN_N_STAR", "3001"))
    training_iterations = int(os.getenv("PINN_EPOCHS", "80000"))
    robustness_iterations = int(
        os.getenv("PINN_ROBUSTNESS_EPOCHS", str(training_iterations))
    )
    gate_time = float(os.getenv("PINN_GATE_TIME", "50"))
    output_directory = Path(os.getenv("PINN_OUTPUT_DIR", "results"))
    output_directory.mkdir(parents=True, exist_ok=True)

    initial_time = np.array([[0.0]], dtype=np.float32)
    initial_probability = np.array(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    residual_points = build_collocation_points(t_end, nominal_residual_points)
    prediction_time = np.linspace(
        0.0, t_end, prediction_points, dtype=np.float32
    ).reshape(-1, 1)

    print(
        f"Training main PINN with {len(residual_points)} unique residual points "
        f"for {training_iterations} iterations."
    )
    main_pinn = SevenStatePINN(
        initial_time,
        initial_probability,
        residual_points,
        gate_time=gate_time,
        absorbing_failure=False,
    )
    main_pinn.train(training_iterations, make_optimizer())
    state_probabilities = main_pinn.predict(prediction_time)

    print(
        "Training auxiliary first-passage PINN for the robustness index "
        f"for {robustness_iterations} iterations."
    )
    first_passage_pinn = SevenStatePINN(
        initial_time,
        initial_probability,
        residual_points,
        gate_time=gate_time,
        absorbing_failure=True,
    )
    first_passage_pinn.train(robustness_iterations, make_optimizer())
    first_passage_probabilities = first_passage_pinn.predict(prediction_time)

    probability_table = pd.DataFrame(
        state_probabilities, columns=[f"probability_{name}" for name in STATE_NAMES]
    )
    probability_table.insert(0, "time_h", prediction_time[:, 0])

    resilience_table = calculate_resilience_indices(
        prediction_time[:, 0], state_probabilities, first_passage_probabilities
    )
    summary_table = build_summary(resilience_table)

    probability_table.to_csv(
        output_directory / "state_probabilities.csv", index=False
    )
    resilience_table.to_csv(
        output_directory / "resilience_indices.csv", index=False
    )
    pd.DataFrame(main_pinn.loss_history).to_csv(
        output_directory / "main_pinn_training_loss.csv", index=False
    )
    pd.DataFrame(first_passage_pinn.loss_history).to_csv(
        output_directory / "robustness_pinn_training_loss.csv", index=False
    )
    summary_table.to_csv(output_directory / "resilience_summary.csv", index=False)

    print("\nFinal resilience results")
    print(summary_table.to_string(index=False))
    print(f"\nResults saved to: {output_directory.resolve()}")


if __name__ == "__main__":
    main()
