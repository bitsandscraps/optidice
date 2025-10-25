import tensorflow as tf
import tensorflow_probability as tfp


def _normalize_spec(spec):
    """Convert various spec-like objects into tf.TensorSpec instances."""
    if isinstance(spec, tf.TensorSpec):
        return spec

    if (
        isinstance(spec, tf.TypeSpec)
        and hasattr(spec, "shape")
        and hasattr(spec, "dtype")
    ):
        return tf.TensorSpec(shape=spec.shape, dtype=spec.dtype)

    if hasattr(spec, "shape") and hasattr(spec, "dtype"):
        # Support objects with TensorSpec-like attributes (e.g., tf_agents specs).
        return tf.TensorSpec(shape=tuple(spec.shape), dtype=spec.dtype)

    if isinstance(spec, (tuple, list)):
        return tf.TensorSpec(shape=tuple(spec), dtype=tf.float32)

    if isinstance(spec, int):
        return tf.TensorSpec(shape=(spec,), dtype=tf.float32)

    raise ValueError(f"Unsupported spec type: {type(spec)}")


def _dummy_input_from_spec(spec):
    spec = _normalize_spec(spec)
    shape = tuple(dim if dim is not None else 1 for dim in spec.shape)
    dtype = spec.dtype if spec.dtype is not None else tf.float32
    return tf.zeros((1,) + shape, dtype=dtype)


def _mlp_layers(hidden_sizes, activation_fn, kernel_initializer, name_prefix):
    kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    layers = []
    for index, hidden_size in enumerate(hidden_sizes):
        layers.append(
            tf.keras.layers.Dense(
                hidden_size,
                activation=activation_fn,
                kernel_initializer=kernel_initializer,
                name=f"{name_prefix}_{index}",
            )
        )
    return layers


class _BaseNetwork(tf.keras.Model):

    def __init__(self, input_tensor_spec, name=None):
        super().__init__(name=name)
        if isinstance(input_tensor_spec, (tuple, list)):
            self._input_specs = tuple(
                _normalize_spec(spec) for spec in input_tensor_spec
            )
        else:
            self._input_specs = (_normalize_spec(input_tensor_spec),)

    def create_variables(self):
        """Backwards-compatible variable creation helper."""
        dummy_inputs = tuple(_dummy_input_from_spec(spec) for spec in self._input_specs)
        # Ensure variables are created by running a forward pass.
        self(dummy_inputs, training=False)

    @staticmethod
    def _ensure_tuple_inputs(inputs):
        if isinstance(inputs, (tuple, list)):
            return tuple(inputs)
        return (inputs,)


class ValueNetwork(_BaseNetwork):
    def __init__(
        self,
        input_tensor_spec,
        hidden_sizes,
        output_activation_fn=None,
        last_layer_bias=None,
        output_dim=None,
        name="ValueNetwork",
    ):
        """Create an instance of `ValueNetwork`."""
        super(ValueNetwork, self).__init__(input_tensor_spec, name=name)

        self._output_dim = output_dim
        self._fc_layers = _mlp_layers(
            hidden_sizes,
            activation_fn=tf.nn.relu,
            kernel_initializer="glorot_uniform",
            name_prefix="mlp",
        )
        last_layer_initializer = tf.keras.initializers.RandomUniform(-3e-3, 3e-3)
        self._last_layer = tf.keras.layers.Dense(
            output_dim or 1,
            activation=output_activation_fn,
            kernel_initializer=last_layer_initializer,
            bias_initializer=last_layer_bias or last_layer_initializer,
            name="value",
        )

    def call(self, inputs, training=False):
        inputs = self._ensure_tuple_inputs(inputs)
        h = tf.concat(inputs, axis=-1)
        for layer in self._fc_layers:
            h = layer(h, training=training)
        h = self._last_layer(h)

        if self._output_dim is None:
            h = tf.reshape(h, [-1])

        return h, ()


class TanhNormalPolicy(_BaseNetwork):

    def __init__(
        self,
        input_tensor_spec,
        action_dim,
        hidden_sizes,
        name="TanhNormalPolicy",
        mean_range=(-7.0, 7.0),
        logstd_range=(-5.0, 2.0),
        eps=1e-6,
    ):
        super(TanhNormalPolicy, self).__init__(input_tensor_spec, name=name)

        self._action_dim = action_dim

        self._fc_layers = _mlp_layers(
            hidden_sizes,
            activation_fn=tf.nn.relu,
            kernel_initializer="glorot_uniform",
            name_prefix="mlp",
        )
        last_layer_initializer = tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
        self._fc_mean = tf.keras.layers.Dense(
            action_dim,
            name="policy_mean_dense",
            kernel_initializer=last_layer_initializer,
            bias_initializer=last_layer_initializer,
        )
        self._fc_logstd = tf.keras.layers.Dense(
            action_dim,
            name="policy_logstd_dense",
            kernel_initializer=last_layer_initializer,
            bias_initializer=last_layer_initializer,
        )

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def call(self, inputs, training=False):
        inputs = self._ensure_tuple_inputs(inputs)
        h = tf.concat(inputs, axis=-1)
        for layer in self._fc_layers:
            h = layer(h, training=training)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        logstd = self._fc_logstd(h)
        logstd = tf.clip_by_value(logstd, self.logstd_min, self.logstd_max)
        std = tf.exp(logstd)
        pretanh_action_dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean, scale_diag=std
        )
        pretanh_action = pretanh_action_dist.sample()
        action = tf.tanh(pretanh_action)
        log_prob, pretanh_log_prob = self.log_prob(
            pretanh_action_dist, pretanh_action, is_pretanh_action=True
        )

        return (
            action,
            pretanh_action,
            log_prob,
            pretanh_log_prob,
            pretanh_action_dist,
        ), ()

    def log_prob(self, pretanh_action_dist, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = tf.tanh(pretanh_action)
        else:
            pretanh_action = tf.atanh(
                tf.clip_by_value(action, -1 + self.eps, 1 - self.eps)
            )

        pretanh_log_prob = pretanh_action_dist.log_prob(pretanh_action)
        log_prob = pretanh_log_prob - tf.reduce_sum(
            tf.math.log(1 - action**2 + self.eps), axis=-1
        )

        return log_prob, pretanh_log_prob

    def deterministic_action(self, inputs):
        inputs = self._ensure_tuple_inputs(inputs)
        h = tf.concat(inputs, axis=-1)
        for layer in self._fc_layers:
            h = layer(h, training=False)

        mean = self._fc_mean(h)
        mean = tf.clip_by_value(mean, self.mean_min, self.mean_max)
        action = tf.tanh(mean)

        return action


class TanhMixtureNormalPolicy(_BaseNetwork):

    def __init__(
        self,
        input_tensor_spec,
        action_dim,
        hidden_sizes,
        num_components=2,
        name="TanhMixtureNormalPolicy",
        mean_range=(-9.0, 9.0),
        logstd_range=(-5.0, 2.0),
        eps=1e-6,
        mdn_temperature=1.0,
    ):
        super(TanhMixtureNormalPolicy, self).__init__(input_tensor_spec, name=name)

        self._action_dim = action_dim
        self._num_components = num_components
        self._mdn_temperature = mdn_temperature

        self._fc_layers = _mlp_layers(
            hidden_sizes,
            activation_fn=tf.nn.relu,
            kernel_initializer="glorot_uniform",
            name_prefix="mlp",
        )
        last_layer_initializer = tf.keras.initializers.RandomUniform(-1e-3, 1e-3)
        self._fc_means = tf.keras.layers.Dense(
            num_components * action_dim,
            name="policy_mean_dense",
            kernel_initializer="glorot_uniform",
            bias_initializer=last_layer_initializer,
        )
        self._fc_logstds = tf.keras.layers.Dense(
            num_components * action_dim,
            name="policy_logstd_dense",
            kernel_initializer=last_layer_initializer,
            bias_initializer=last_layer_initializer,
        )
        self._fc_logits = tf.keras.layers.Dense(
            num_components,
            name="policy_logits_dense",
            kernel_initializer="glorot_uniform",
            bias_initializer=last_layer_initializer,
        )

        self.mean_min, self.mean_max = mean_range
        self.logstd_min, self.logstd_max = logstd_range
        self.eps = eps

    def call(self, inputs, training=False):
        inputs = self._ensure_tuple_inputs(inputs)
        h = tf.concat(inputs, axis=-1)
        for layer in self._fc_layers:
            h = layer(h, training=training)

        means = self._fc_means(h)
        means = tf.clip_by_value(means, self.mean_min, self.mean_max)
        means = tf.reshape(means, (-1, self._num_components, self._action_dim))
        logstds = self._fc_logstds(h)
        logstds = tf.clip_by_value(logstds, self.logstd_min, self.logstd_max)
        logstds = tf.reshape(logstds, (-1, self._num_components, self._action_dim))
        stds = tf.exp(logstds)

        component_logits = self._fc_logits(h) / self._mdn_temperature

        pretanh_actions_dist = tfp.distributions.MultivariateNormalDiag(
            loc=means, scale_diag=stds
        )
        component_dist = tfp.distributions.Categorical(logits=component_logits)

        pretanh_actions = (
            pretanh_actions_dist.sample()
        )  # (batch_size, num_components, action_dim)
        component = component_dist.sample()  # (batch_size)

        batch_idx = tf.range(tf.shape(inputs[0])[0])
        pretanh_action = tf.gather_nd(
            pretanh_actions, tf.stack([batch_idx, component], axis=1)
        )
        action = tf.tanh(pretanh_action)

        log_prob, pretanh_log_prob = self.log_prob(
            (component_dist, pretanh_actions_dist),
            pretanh_action,
            is_pretanh_action=True,
        )

        return (
            action,
            pretanh_action,
            log_prob,
            pretanh_log_prob,
            (component_dist, pretanh_actions_dist),
        ), ()

    def log_prob(self, dists, action, is_pretanh_action=True):
        if is_pretanh_action:
            pretanh_action = action
            action = tf.tanh(pretanh_action)
        else:
            pretanh_action = tf.atanh(
                tf.clip_by_value(action, -1 + self.eps, 1 - self.eps)
            )

        component_dist, pretanh_actions_dist = dists
        logits_fn = getattr(component_dist, "logits_parameter", None)
        component_logits = logits_fn() if callable(logits_fn) else component_dist.logits
        component_log_prob = component_logits - tf.math.reduce_logsumexp(
            component_logits, axis=-1, keepdims=True
        )

        pretanh_actions = tf.tile(
            pretanh_action[:, None, :], (1, self._num_components, 1)
        )  # (batch_size, num_components, action_dim)

        pretanh_log_prob = tf.reduce_logsumexp(
            component_log_prob + pretanh_actions_dist.log_prob(pretanh_actions), axis=1
        )
        log_prob = pretanh_log_prob - tf.reduce_sum(
            tf.math.log(1 - action**2 + self.eps), axis=-1
        )

        return log_prob, pretanh_log_prob
