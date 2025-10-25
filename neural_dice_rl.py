import json
import os
import time

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import util
from networks import TanhMixtureNormalPolicy, TanhNormalPolicy, ValueNetwork

np.set_printoptions(precision=3, suppress=True)


class NeuralDICERL(tf.keras.layers.Layer):
    """Offline policy Optimization via Stationary DIstribution Correction Estimation (OptiDICE)"""

    def __init__(self, observation_spec, action_spec, config):
        super(NeuralDICERL, self).__init__()

        self._gamma = config["gamma"]
        self._policy_extraction = config["policy_extraction"]
        self._env_name = config["env_name"]
        self._total_iterations = config["total_iterations"]
        self._warmup_iterations = config["warmup_iterations"]
        self._data_policy = config["data_policy"]
        self._data_policy_num_mdn_components = config["data_policy_num_mdn_components"]
        self._data_policy_mdn_temperature = config["data_policy_mdn_temperature"]
        self._use_policy_entropy_constraint = config["use_policy_entropy_constraint"]
        self._use_data_policy_entropy_constraint = config[
            "use_data_policy_entropy_constraint"
        ]
        self._target_entropy = config["target_entropy"]
        self._hidden_sizes = config["hidden_sizes"]
        self._batch_size = config["batch_size"]
        self._alpha = config["alpha"]
        self._f = config["f"]
        self._gendice_v = config["gendice_v"]
        self._gendice_e = config["gendice_e"]
        self._gendice_loss_type = config["gendice_loss_type"]
        self._lr = config["lr"]
        self._e_loss_type = config["e_loss_type"]
        self._v_l2_reg = config["v_l2_reg"]
        self._lamb_scale = config["lamb_scale"]

        self._iteration = tf.Variable(
            0, dtype=tf.int64, name="iteration", trainable=False
        )
        self._optimizers = dict()

        # create networks / variables for DICE-RL
        self._v_network = ValueNetwork(
            (observation_spec,),
            hidden_sizes=self._hidden_sizes,
            output_activation_fn=None,
            name="v",
        )
        self._v_network.create_variables()
        self._optimizers["v"] = tf.keras.optimizers.Adam(self._lr)

        self._e_network = ValueNetwork(
            (observation_spec, action_spec), hidden_sizes=self._hidden_sizes, name="e"
        )
        self._e_network.create_variables()
        self._optimizers["e"] = tf.keras.optimizers.Adam(self._lr)

        # GenDICE regularization, i.e., E[w] = 1.
        if self._gendice_v:
            self._lamb_v = tf.Variable(0.0, dtype=tf.float32, name="lamb_v")
            self._optimizers["lamb_v"] = tf.keras.optimizers.Adam(self._lr)
        else:
            self._lamb_v = 0

        if self._gendice_e:
            self._lamb_e = tf.Variable(0.0, dtype=tf.float32, name="lamb_e")
            self._optimizers["lamb_e"] = tf.keras.optimizers.Adam(self._lr)
        else:
            self._lamb_e = 0

        # f-divergence functions
        # NOTE: g(x) = f(ReLU((f')^{-1}(x)))
        # NOTE: r(x) = ReLU((f')^{-1}(x)
        if self._f == "chisquare":
            self._f_fn = lambda x: 0.5 * (x - 1) ** 2
            self._f_prime_inv_fn = lambda x: x + 1
            self._g_fn = lambda x: 0.5 * (tf.nn.relu(x + 1) - 1) ** 2
            self._r_fn = lambda x: tf.nn.relu(self._f_prime_inv_fn(x))
            self._log_r_fn = lambda x: tf.where(
                x < 0, tf.math.log(1e-10), tf.math.log(tf.maximum(x, 0) + 1)
            )
        elif self._f == "kl":
            self._f_fn = lambda x: x * tf.math.log(x + 1e-10)
            self._f_prime_inv_fn = lambda x: tf.math.exp(x - 1)
            self._g_fn = lambda x: tf.math.exp(x - 1) * (x - 1)
            self._r_fn = lambda x: self._f_prime_inv_fn(x)
            self._log_r_fn = lambda x: x - 1
        elif self._f == "elu":
            self._f_fn = lambda x: tf.where(
                x < 1, x * (tf.math.log(x + 1e-10) - 1) + 1, 0.5 * (x - 1) ** 2
            )
            self._f_prime_inv_fn = lambda x: tf.where(
                x < 0, tf.math.exp(tf.minimum(x, 0)), x + 1
            )
            self._g_fn = lambda x: tf.where(
                x < 0,
                tf.math.exp(tf.minimum(x, 0)) * (tf.minimum(x, 0) - 1) + 1,
                0.5 * x**2,
            )
            self._r_fn = lambda x: self._f_prime_inv_fn(x)
            self._log_r_fn = lambda x: tf.where(
                x < 0, x, tf.math.log(tf.maximum(x, 0) + 1)
            )
        else:
            raise NotImplementedError()

        # policy
        self._policy_network = TanhNormalPolicy(
            (observation_spec,),
            action_spec.shape[0],
            hidden_sizes=self._hidden_sizes,
            mean_range=config["mean_range"],
            logstd_range=config["logstd_range"],
        )
        self._policy_network.create_variables()
        self._optimizers["policy"] = tf.keras.optimizers.Adam(self._lr)

        if self._use_policy_entropy_constraint:
            self._log_ent_coeff = tf.Variable(0.0, dtype=tf.float32, name="ent_coeff")
            self._optimizers["ent_coeff"] = tf.keras.optimizers.Adam(self._lr)

        # data policy
        if self._policy_extraction == "iproj":
            if config["data_policy"] == "tanh_normal":
                self._data_policy_network = TanhNormalPolicy(
                    (observation_spec,),
                    action_spec.shape[0],
                    hidden_sizes=self._hidden_sizes,
                    mean_range=config["data_mean_range"],
                    logstd_range=config["data_logstd_range"],
                )
            elif config["data_policy"] == "tanh_mdn":
                if config["data_policy_num_mdn_components"] == 1:
                    self._data_policy_network = TanhNormalPolicy(
                        (observation_spec,),
                        action_spec.shape[0],
                        hidden_sizes=self._hidden_sizes,
                        mean_range=config["data_mean_range"],
                        logstd_range=config["data_logstd_range"],
                    )
                else:
                    self._data_policy_network = TanhMixtureNormalPolicy(
                        (observation_spec,),
                        action_spec.shape[0],
                        hidden_sizes=self._hidden_sizes,
                        mean_range=config["data_mean_range"],
                        logstd_range=config["data_logstd_range"],
                        num_components=config["data_policy_num_mdn_components"],
                        mdn_temperature=config["data_policy_mdn_temperature"],
                    )
            else:
                raise NotImplementedError(
                    f'unknown data policy={config["data_policy"]}'
                )
            self._data_policy_network.create_variables()
            self._optimizers["data_policy"] = tf.keras.optimizers.Adam(self._lr)

            if self._use_data_policy_entropy_constraint:
                self._data_log_ent_coeff = tf.Variable(
                    0.0, dtype=tf.float32, name="data_ent_coeff"
                )
                self._optimizers["data_ent_coeff"] = tf.keras.optimizers.Adam(self._lr)

    def v_loss(self, initial_v_values, e_v, w_v, f_w_v, result={}):
        # Compute v loss
        v_loss0 = (1 - self._gamma) * tf.reduce_mean(initial_v_values)
        v_loss1 = tf.reduce_mean(-self._alpha * f_w_v)
        if self._gendice_loss_type == "gendice":
            v_loss2 = tf.reduce_mean(w_v * (e_v - self._lamb_scale * self._lamb_v))
            v_loss3 = self._lamb_scale * (self._lamb_v + self._lamb_v**2 / 2)
        elif self._gendice_loss_type == "bestdice":
            v_loss2 = tf.reduce_mean(w_v * (e_v - self._lamb_v))
            v_loss3 = self._lamb_v
        else:
            raise NotImplementedError
        v_loss = v_loss0 + v_loss1 + v_loss2 + v_loss3

        v_l2_norm = tf.linalg.global_norm(self._v_network.variables)

        if self._v_l2_reg is not None:
            v_loss += self._v_l2_reg * v_l2_norm

        result.update(
            {
                "loss/V0": v_loss0,
                "loss/V1": v_loss1,
                "loss/V2": v_loss2,
                "loss/V3": v_loss3,
                "loss/V": v_loss,
                "loss/Vnorm": v_l2_norm,
            }
        )

        return result

    def lamb_v_loss(self, e_v, w_v, f_w_v, result={}):
        # GenDICE regularization: E_D[w(s,a)] = 1
        if self._gendice_loss_type == "gendice":
            lamb_v_loss = tf.reduce_mean(
                -self._alpha * f_w_v
                + w_v * (e_v - self._lamb_scale * self._lamb_v)
                + self._lamb_scale * (self._lamb_v + self._lamb_v**2 / 2)
            )
        elif self._gendice_loss_type == "bestdice":
            lamb_v_loss = tf.reduce_mean(
                -self._alpha * f_w_v
                + w_v * (e_v - self._lamb_scale * self._lamb_v)
                + self._lamb_v
            )
        else:
            raise NotImplementedError

        result.update({"loss/LambdaV": lamb_v_loss, "train/LambdaV": self._lamb_v})

        return result

    def e_loss(self, e_v, e_values, w_e, f_w_e, result={}):
        # Compute e loss
        if self._e_loss_type == "minimax":
            e_loss = tf.reduce_mean(
                self._alpha * f_w_e - w_e * (e_v - self._lamb_scale * self._lamb_e)
            )
        elif self._e_loss_type == "mse":
            e_loss = tf.reduce_mean((e_v - e_values) ** 2)
        else:
            raise NotImplementedError

        e_l2_norm = tf.linalg.global_norm(self._e_network.variables)

        result.update({"loss/E": e_loss, "loss/Enorm": e_l2_norm})

        return result

    def lamb_e_loss(self, e_v, w_e, f_w_e, result={}):
        # GenDICE regularization: E_D[w(s,a)] = 1
        if self._gendice_loss_type == "gendice":
            lamb_e_loss = tf.reduce_mean(
                -self._alpha * f_w_e
                + w_e * (e_v - self._lamb_scale * self._lamb_e)
                + self._lamb_scale * (self._lamb_e + self._lamb_e**2 / 2)
            )
        elif self._gendice_loss_type == "bestdice":
            lamb_e_loss = tf.reduce_mean(
                -self._alpha * f_w_e
                + w_e * (e_v - self._lamb_scale * self._lamb_e)
                + self._lamb_e
            )
        else:
            raise NotImplementedError

        result.update({"loss/LambdaE": lamb_e_loss, "train/LambdaE": self._lamb_e})

        return result

    def policy_loss(self, observation, action, w_e, result={}):
        # Compute policy loss
        (
            sampled_action,
            sampled_pretanh_action,
            sampled_action_log_prob,
            sampled_pretanh_action_log_prob,
            pretanh_action_dist,
        ), _ = self._policy_network((observation,))
        # Entropy is estimated on newly sampled action.
        negative_entropy_loss = tf.reduce_mean(sampled_action_log_prob)

        policy_l2_norm = tf.linalg.global_norm(self._policy_network.variables)

        if self._policy_extraction == "wbc":
            # Weighted BC
            action_log_prob, _ = self._policy_network.log_prob(
                pretanh_action_dist, action, is_pretanh_action=False
            )
            policy_loss = -tf.reduce_mean(w_e * action_log_prob)

        elif self._policy_extraction == "iproj":
            # Information projection
            (_, _, _, _, data_pretanh_action_dist), _ = self._data_policy_network(
                (observation,)
            )

            sampled_e_values, _ = self._e_network((observation, sampled_action))
            if self._gendice_loss_type == "gendice":
                sampled_log_w_e = self._log_r_fn(
                    (sampled_e_values - self._lamb_scale * self._lamb_e) / self._alpha
                )
            elif self._gendice_loss_type == "bestdice":
                sampled_log_w_e = self._log_r_fn(
                    (sampled_e_values - self._lamb_e) / self._alpha
                )
            else:
                raise NotImplementedError()

            _, sampled_pretanh_action_data_log_prob = (
                self._data_policy_network.log_prob(
                    data_pretanh_action_dist, sampled_pretanh_action
                )
            )
            kl = sampled_pretanh_action_log_prob - sampled_pretanh_action_data_log_prob
            policy_loss = -tf.reduce_mean(sampled_log_w_e - kl)

            result.update({"loss/KL": tf.reduce_mean(kl)})

        else:
            raise NotImplementedError()

        if self._use_policy_entropy_constraint:
            ent_coeff = tf.exp(self._log_ent_coeff)
            policy_loss += ent_coeff * negative_entropy_loss

            ent_coeff_loss = -self._log_ent_coeff * (
                sampled_action_log_prob + self._target_entropy
            )

            result.update(
                {
                    "loss/EntCoeff": tf.reduce_mean(ent_coeff_loss),
                    "train/EntCoeff": ent_coeff,
                }
            )

        result.update(
            {
                "loss/Policy": policy_loss,
                "loss/PolicyNorm": policy_l2_norm,
                "loss/NegEnt": negative_entropy_loss,
            }
        )

        return result

    def data_policy_loss(self, observation, action, result={}):
        # Compute data policy loss
        (_, _, data_sampled_action_log_prob, _, data_policy_dists), _ = (
            self._data_policy_network((observation,))
        )

        data_action_log_prob, _ = self._data_policy_network.log_prob(
            data_policy_dists, action, is_pretanh_action=False
        )
        data_policy_loss = -tf.reduce_mean(data_action_log_prob)

        # Entropy is estimated on newly sampled action.
        data_negative_entropy_loss = tf.reduce_mean(data_sampled_action_log_prob)

        if self._use_data_policy_entropy_constraint:
            data_ent_coeff = tf.exp(self._data_log_ent_coeff)
            data_policy_loss += data_ent_coeff * data_negative_entropy_loss

            data_ent_coeff_loss = -self._data_log_ent_coeff * (
                data_sampled_action_log_prob + self._target_entropy
            )

            result.update(
                {
                    "loss/data/EntCoeff": tf.reduce_mean(data_ent_coeff_loss),
                    "train/data/EntCoeff": data_ent_coeff,
                }
            )

        result.update(
            {
                "loss/data/NegEnt": data_negative_entropy_loss,
                "loss/data/Policy": data_policy_loss,
            }
        )

        return result

    @tf.function
    def train_step(
        self,
        initial_observation,
        observation,
        action,
        reward,
        next_observation,
        terminal,
    ):
        with tf.GradientTape(persistent=True) as tape:

            # Shared network values
            initial_v_values, _ = self._v_network((initial_observation,))
            v_values, _ = self._v_network((observation,))
            next_v_values, _ = self._v_network((next_observation,))

            e_v = reward + (1 - terminal) * self._gamma * next_v_values - v_values
            preactivation_v = (e_v - self._lamb_scale * self._lamb_v) / self._alpha
            w_v = self._r_fn(preactivation_v)
            f_w_v = self._g_fn(preactivation_v)

            e_values, _ = self._e_network((observation, action))
            preactivation_e = (e_values - self._lamb_scale * self._lamb_e) / self._alpha
            w_e = self._r_fn(preactivation_e)
            f_w_e = self._g_fn(preactivation_e)

            # compute loss within GradientTape context manager
            loss_result = self.v_loss(initial_v_values, e_v, w_v, f_w_v, result={})
            if self._gendice_v:
                loss_result = self.lamb_v_loss(e_v, w_v, f_w_v, result=loss_result)

            loss_result = self.e_loss(
                tf.stop_gradient(e_v), e_values, w_e, f_w_e, result=loss_result
            )
            if self._gendice_e:
                loss_result = self.lamb_e_loss(
                    tf.stop_gradient(e_v), w_e, f_w_e, result=loss_result
                )

            loss_result = self.policy_loss(
                observation, action, tf.stop_gradient(w_e), result=loss_result
            )
            if self._policy_extraction == "iproj":
                loss_result = self.data_policy_loss(
                    observation, action, result=loss_result
                )

        v_grads = tape.gradient(loss_result["loss/V"], self._v_network.variables)
        self._optimizers["v"].apply_gradients(zip(v_grads, self._v_network.variables))

        if self._gendice_v:
            lamb_v_grads = tape.gradient(loss_result["loss/LambdaV"], [self._lamb_v])
            self._optimizers["lamb_v"].apply_gradients(
                zip(lamb_v_grads, [self._lamb_v])
            )

        e_grads = tape.gradient(loss_result["loss/E"], self._e_network.variables)
        self._optimizers["e"].apply_gradients(zip(e_grads, self._e_network.variables))

        if self._gendice_e:
            lamb_e_grads = tape.gradient(loss_result["loss/LambdaE"], [self._lamb_e])
            self._optimizers["lamb_e"].apply_gradients(
                zip(lamb_e_grads, [self._lamb_e])
            )

        policy_grads = tape.gradient(
            loss_result["loss/Policy"], self._policy_network.variables
        )

        def _apply_policy():
            self._optimizers["policy"].apply_gradients(
                zip(policy_grads, self._policy_network.variables)
            )
            return tf.constant(0)

        tf.cond(
            self._iteration >= self._warmup_iterations,
            _apply_policy,
            lambda: tf.constant(0),
        )

        if self._use_policy_entropy_constraint:
            ent_coeff_grads = tape.gradient(
                loss_result["loss/EntCoeff"], [self._log_ent_coeff]
            )

            def _apply_ent_coeff():
                self._optimizers["ent_coeff"].apply_gradients(
                    zip(ent_coeff_grads, [self._log_ent_coeff])
                )
                return tf.constant(0)

            tf.cond(
                self._iteration >= self._warmup_iterations,
                _apply_ent_coeff,
                lambda: tf.constant(0),
            )

        if self._policy_extraction == "iproj":
            data_policy_grads = tape.gradient(
                loss_result["loss/data/Policy"], self._data_policy_network.variables
            )
            self._optimizers["data_policy"].apply_gradients(
                zip(data_policy_grads, self._data_policy_network.variables)
            )

            if (
                self._data_policy in ["tanh_normal", "tanh_mdn"]
                and self._use_data_policy_entropy_constraint
            ):
                data_ent_coeff_grads = tape.gradient(
                    loss_result["loss/data/EntCoeff"], [self._data_log_ent_coeff]
                )
                self._optimizers["data_ent_coeff"].apply_gradients(
                    zip(data_ent_coeff_grads, [self._data_log_ent_coeff])
                )

        self._iteration.assign_add(1)

        return loss_result

    @tf.function
    def step(self, observation):
        """
        observation: batch_size x obs_dim
        """
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        action = self._policy_network.deterministic_action((observation,))

        return action, None


def run(config):
    np.random.seed(config["seed"])
    tf.random.set_seed(config["seed"])

    log_parent = config.get("log_dir") or "runs"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    summary_dir = os.path.join(log_parent, f"{config['env_name']}_{timestamp}")
    os.makedirs(summary_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(summary_dir)
    with summary_writer.as_default():
        tf.summary.text(
            "config",
            tf.convert_to_tensor(json.dumps(config, default=str, indent=2)),
            step=0,
        )
        tf.summary.flush()

    # load dataset
    env = gym.make(config["env_name"])
    env.seed(config["seed"])
    initial_obs_dataset, dataset, dataset_statistics = util.dice_dataset(
        env,
        standardize_observation=True,
        absorbing_state=config["absorbing_state"],
        standardize_reward=config["standardize_reward"],
    )
    if (
        config["use_policy_entropy_constraint"]
        or config["use_data_policy_entropy_constraint"]
    ):
        if config["target_entropy"] is None:
            config["target_entropy"] = -np.prod(env.action_space.shape)

    def _sample_minibatch(batch_size, reward_scale):
        initial_indices = np.random.randint(
            0, dataset_statistics["N_initial_observations"], batch_size
        )
        indices = np.random.randint(0, dataset_statistics["N"], batch_size)
        sampled_dataset = (
            initial_obs_dataset["initial_observations"][initial_indices],
            dataset["observations"][indices],
            dataset["actions"][indices],
            dataset["rewards"][indices] * reward_scale,
            dataset["next_observations"][indices],
            dataset["terminals"][indices],
        )
        return tuple(map(tf.convert_to_tensor, sampled_dataset))

    # Create an agent
    obs_dim = (
        (dataset_statistics["observation_dim"] + 1)
        if config["absorbing_state"]
        else dataset_statistics["observation_dim"]
    )
    observation_spec = tf.TensorSpec(
        shape=(obs_dim,), dtype=tf.float32, name="observation"
    )
    action_spec = tf.TensorSpec(
        shape=(dataset_statistics["action_dim"],), dtype=tf.float32, name="action"
    )

    agent = NeuralDICERL(
        observation_spec=observation_spec, action_spec=action_spec, config=config
    )

    start_iteration = 0

    # Start training
    last_start_time = time.time()
    for iteration in tqdm(
        range(start_iteration, config["total_iterations"] + 1),
        desc="DICE",
        initial=start_iteration,
        ascii=True,
    ):
        # Sample mini-batch data from dataset
        initial_observation, observation, action, reward, next_observation, terminal = (
            _sample_minibatch(config["batch_size"], config["reward_scale"])
        )

        # Perform gradient descent
        train_result = agent.train_step(
            initial_observation, observation, action, reward, next_observation, terminal
        )
        if iteration % config["log_iterations"] == 0:
            train_result = {k: v.numpy() for k, v in train_result.items()}
            if iteration >= config["warmup_iterations"]:
                # evaluation via real-env rollout
                eval = util.evaluate(
                    env,
                    agent,
                    dataset_statistics,
                    absorbing_state=config["absorbing_state"],
                    pid=config.get("pid"),
                    print_fn=None,
                    show_progress=False,
                )
                train_result["eval/mean"] = np.mean(eval)
                train_result["eval/std"] = np.std(eval)

            with summary_writer.as_default():
                for key, value in train_result.items():
                    tf.summary.scalar(key, value, step=iteration)


if __name__ == "__main__":
    from default_config import get_parser

    args = get_parser().parse_args()
    run(vars(args))
