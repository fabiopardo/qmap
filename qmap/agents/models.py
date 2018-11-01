import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim


def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        inpt = tf.cast(inpt, tf.float32)
        inpt = tf.div(inpt, 255.)
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def ConvMlp(convs, hiddens, dueling=False, layer_norm=False):
    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)


class ConvDeconvMap(object):
    def __init__(self, convs, middle_hiddens, deconvs, coords_shape, dueling, layer_norm, activation_fn):
        self.description = 'ConvDeconvMap-' + str(convs + middle_hiddens + deconvs) + '-' + activation_fn.__repr__().split(' ')[1]
        self.description = self.description.replace(' ', '')
        if dueling: self.description += '-duel'
        if layer_norm: self.description += '-norm'

        def call(inpt, n_actions, scope, reuse=False):
            coords_size = coords_shape[0] * coords_shape[1]
            batch_size = tf.shape(inpt)[0]
            print('~~~~~~~~~~')
            print('NETWORK:')
            inpt = tf.cast(inpt, tf.float32)
            inpt = tf.div(inpt, 255.)
            print(inpt)

            with tf.variable_scope(scope, reuse=reuse):
                print('ENCODER')
                encoder_out = inpt
                with tf.variable_scope('encoder'):
                    for num_outputs, kernel_size, stride in convs:
                        encoder_out = layers.conv2d(encoder_out, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, activation_fn=None)
                        # if layer_norm:
                        #     encoder_out = layers.layer_norm(encoder_out, center=True, scale=True)
                        encoder_out = activation_fn(encoder_out)
                        print(encoder_out)

                with tf.variable_scope('middle_hiddens'):
                    print('MIDDLE')
                    middle_out = encoder_out

                    if len(middle_hiddens) != 0:
                        encoded_shape = tf.shape(middle_out)
                        middle_out = layers.flatten(middle_out)
                        print(middle_out)

                        for hidden in middle_hiddens + [middle_out.shape.as_list()[1]]:
                            middle_out = layers.fully_connected(middle_out, num_outputs=hidden, activation_fn=None)
                            if layer_norm:
                                middle_out = layers.layer_norm(middle_out, center=True, scale=True)
                            middle_out = activation_fn(middle_out)
                            print(middle_out)

                        middle_out = tf.reshape(middle_out, encoded_shape)
                        print(middle_out)

                with tf.variable_scope('action_value'):
                    print('DECODER Q')
                    action_scores = middle_out
                    for i, (num_outputs, kernel_size, stride) in enumerate(deconvs):
                        if i == len(deconvs)-1: deconv_activation_fn = None
                        else: deconv_activation_fn = activation_fn
                        action_scores = layers.conv2d_transpose(action_scores, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, activation_fn=None)
                        # if layer_norm:
                        #     action_scores = layers.layer_norm(action_scores, center=True, scale=True)
                        if deconv_activation_fn is not None:
                            action_scores = deconv_activation_fn(action_scores)
                        else:
                            print('last activation function is None :)')
                        print(action_scores)

                if dueling:
                    with tf.variable_scope('state_value'):
                        print('DECODER V')
                        state_score = middle_out
                        print(state_score)
                        for i, (num_outputs, kernel_size, stride) in enumerate(deconvs):
                            if i == len(deconvs)-1:
                                deconv_activation_fn = None
                                num_outputs = 1
                            else: deconv_activation_fn = activation_fn
                            state_score = layers.conv2d_transpose(state_score, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, activation_fn=None)
                            # if layer_norm:
                            #     state_score = layers.layer_norm(state_score, center=True, scale=True)
                            if deconv_activation_fn is not None:
                                state_score = deconv_activation_fn(state_score)
                            else:
                                print('last activation function is None :)')
                            print(state_score)
                    action_scores -= tf.reduce_mean(action_scores, 3, keepdims=True)
                    q_out = state_score + action_scores
                else:
                    q_out = action_scores
                print('OUT')
                print(q_out)

            return q_out

        self.call = call

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class MlpMap(object):
    def __init__(self, hiddens, coords_shape, dueling, layer_norm, activation_fn):
        self.description = 'MlpMap-' + str(hiddens) + '-' + activation_fn.__repr__().split(' ')[1]
        self.description = self.description.replace(' ', '')
        if dueling: self.description += '-duel'
        if layer_norm: self.description += '-norm'

        def call(inpt, n_actions, scope, reuse=False):
            coords_size = coords_shape[0] * coords_shape[1]
            batch_size = tf.shape(inpt)[0]
            print('~~~~~~~~~~')
            print('NETWORK:')
            inpt = tf.cast(inpt, tf.float32)
            inpt = tf.div(inpt, 255.)
            print(inpt)

            with tf.variable_scope(scope, reuse=reuse):
                out = inpt
                with tf.variable_scope('hiddens'):
                    print('HIDDENS')
                    rows, cols, channels = out.get_shape().as_list()[1:]
                    out = layers.flatten(out)
                    print(out)

                    for hidden in hiddens:
                        out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            out = layers.layer_norm(out, center=True, scale=True)
                        out = activation_fn(out)
                        print(out)

                    action_scores = out
                    action_scores = layers.fully_connected(action_scores, num_outputs=rows*cols*n_actions, activation_fn=None)
                    if layer_norm:
                        action_scores = layers.layer_norm(action_scores, center=True, scale=True)
                    print(action_scores)
                    action_scores = tf.reshape(action_scores, (-1, rows, cols, n_actions))
                    print(action_scores)

                if dueling:
                    with tf.variable_scope('state_value'):
                        print('DECODER V')
                        state_score = out
                    state_score = layers.fully_connected(state_score, num_outputs=rows*cols*1, activation_fn=None)
                    if layer_norm:
                        state_score = layers.layer_norm(state_score, center=True, scale=True)
                    print(state_score)
                    state_score = tf.reshape(state_score, (-1, rows, cols, 1))
                    print(state_score)
                    action_scores -= tf.reduce_mean(action_scores, 3, keepdims=True)
                    q_out = state_score + action_scores
                else:
                    q_out = action_scores
                print('OUT')
                print(q_out)

            return q_out

        self.call = call

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)


class ConvDenseQLearner(object):
    def __init__(self, convs, hiddens, dueling, layer_norm, activation_fn):
        self.description = 'ConvDenseDQN-' + str(convs + hiddens) + '-' + activation_fn.__repr__().split(' ')[1]
        if dueling: self.description += '-duel'
        if layer_norm: self.description += '-norm'

        def call(inpt, n_actions, scope, reuse=False):
            batch_size = tf.shape(inpt)[0]
            inpt = tf.cast(inpt, tf.float32)
            inpt = tf.div(inpt, 255.)
            print(inpt)

            with tf.variable_scope(scope, reuse=reuse):
                encoder_out = inpt
                with tf.variable_scope('encoder'):
                    for num_outputs, kernel_size, stride in convs:
                        encoder_out = layers.conv2d(encoder_out, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, activation_fn=None)
                        if layer_norm:
                            encoder_out = layers.layer_norm(encoder_out, center=True, scale=True)
                        encoder_out = activation_fn(encoder_out)
                        print(encoder_out)
                    encoder_out = layers.flatten(encoder_out)
                    print(encoder_out)

                with tf.variable_scope('hiddens'):
                    middle_out = encoder_out
                    for hidden in hiddens:
                        middle_out = layers.fully_connected(middle_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            middle_out = layers.layer_norm(middle_out, center=True, scale=True)
                        middle_out = activation_fn(middle_out)
                        print(middle_out)

                with tf.variable_scope('action_value'):
                    action_out = middle_out
                    action_scores = layers.fully_connected(action_out, num_outputs=n_actions, activation_fn=None)
                    print(action_scores)
                    action_scores = tf.reshape(action_scores, (batch_size, n_actions))
                    print(action_scores)

                if dueling:
                    with tf.variable_scope('state_value'):
                        state_out = middle_out
                        state_score = layers.fully_connected(state_out, num_outputs=coords_size, activation_fn=None)
                        print(state_score)
                        state_score = tf.reshape(state_score, (batch_size))
                        print(state_score)
                    action_scores_mean = tf.reduce_mean(action_scores, 1) # TODO: check
                    action_scores_centered = action_scores - action_scores_mean
                    q_out = state_score + action_scores_centered
                else:
                    q_out = action_scores
                print(q_out)

            return q_out

        self.call = call

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
