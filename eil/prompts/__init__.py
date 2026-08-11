"""EIL role-specific prompt builders.

The EIL runtime has one loyal policy role, one blind adversary role, and two
judge roles. The two judges each make scoped subcalls because their inputs are
deliberately different, but all judge calls use the single EIL judge client.
"""
