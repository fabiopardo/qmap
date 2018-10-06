import baselines.common.tf_util as U
from datetime import datetime
from gym.utils import seeding
import numpy as np
import os
import tensorflow as tf
from termcolor import colored


def build_train(observation_space, coords_shape, model, n_actions, optimizer, grad_norm_clip, scope='q_map'):
    with tf.variable_scope(scope):
        ob_shape = observation_space.shape
        observations = tf.placeholder(tf.float32, [None] + list(ob_shape), name='observations')
        actions = tf.placeholder(tf.int32, [None], name='actions')
        target_qs = tf.placeholder(tf.float32, [None] + list(coords_shape), name='targets')
        weights = tf.placeholder(tf.float32, [None], name='weights')

        q_values = model(inpt=observations, n_actions=n_actions, scope='q_func')
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")

        target_q_values = model(inpt=observations, n_actions=n_actions, scope='target_q_func')
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/target_q_func")

        action_masks = tf.expand_dims(tf.expand_dims(tf.one_hot(actions, n_actions), axis=1), axis=1)
        qs_selected = tf.reduce_sum(q_values * action_masks, 3)

        td_errors = 1 * (qs_selected - target_qs) # TODO: coefficient?
        losses = tf.reduce_mean(tf.square(td_errors), [1, 2]) # TODO: find best, was U.huber_loss
        weighted_loss = tf.reduce_mean(weights * losses)

        if grad_norm_clip is not None:
            gradients = optimizer.compute_gradients(weighted_loss, var_list=q_func_vars)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clip), var)
            optimize = optimizer.apply_gradients(gradients)
            grad_norms = [tf.norm(grad) for grad in gradients]
        else:
            optimize = optimizer.minimize(weighted_loss, var_list=q_func_vars)
            grad_norms = None

        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

    errors = tf.reduce_mean(tf.abs(td_errors), [1, 2]) # TODO: try with the losses directly
    compute_q_values = U.function(inputs=[observations], outputs=q_values)
    compute_double_q_values = U.function(inputs=[observations], outputs=[q_values, target_q_values])
    train = U.function(inputs=[observations, actions, target_qs, weights], outputs=errors, updates=[optimize])
    update_target = U.function([], [], updates=[update_target_expr])
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    train_debug = U.function(inputs=[observations, actions, target_qs, weights], outputs=[errors, weighted_loss, grad_norms, trainable_vars], updates=[optimize])

    return compute_q_values, compute_double_q_values, train, update_target, train_debug
