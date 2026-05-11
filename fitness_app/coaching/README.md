"""Experimental heuristic coaching (angles + phases). Not medical advice.

Future extensions:
- **Gold clips**: store JSON `{phase: {joint: {mean, std}}}` from demo videos; score user angles vs distribution.
- **Procrustes**: translate/scale/rotate 2D skeleton to reference per phase; penalize residual joint error.
"""
