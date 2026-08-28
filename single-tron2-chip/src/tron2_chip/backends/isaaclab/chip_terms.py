"""Tensor operations used by Isaac Lab observation and critic managers."""


def hindsight_goal(reference_goal, external_force, compliance):
    """Batched g_hind = g_ref - C*f for diagonal compliance tensors."""
    if reference_goal.shape[-1] != 3 or external_force.shape[-1] != 3:
        raise ValueError("reference goal and external force require final dimension 3")
    if compliance.shape[-1] != 3:
        raise ValueError("diagonal compliance requires final dimension 3")
    return reference_goal - compliance * external_force


def actor_goal(reference_goal, external_force, compliance, training=True):
    return (
        hindsight_goal(reference_goal, external_force, compliance)
        if training else reference_goal
    )

