"""Teacher-student wrapper for Relational Self-Supervised Learning.

This module provides an EMA teacher that updates model parameters
without copying the student's BatchNorm running statistics.
"""

import stable_pretraining as spt
import torch


class ReSSLTeacherStudentWrapper(spt.TeacherStudentWrapper):
    """Teacher-student wrapper with independent BatchNorm statistics.

    ReSSL updates teacher parameters using an exponential moving average
    (EMA) of student parameters while maintaining independent BatchNorm
    running statistics for the teacher.

    Unlike the standard TeacherStudentWrapper, this implementation
    does not synchronize buffers between the student and teacher.
    """

    @torch.no_grad()
    def update_teacher(self):
        """Update teacher parameters using exponential moving averaging.

        Applies the update:

            teacher = momentum * teacher + (1 - momentum) * student

        BatchNorm running statistics and other buffers are left unchanged.

        The update is skipped during evaluation, when the teacher and
        student share parameters, or when EMA momentum equals 1.0.

        Raises:
            RuntimeError: If student and teacher parameter counts differ.
        """
        if not self.training or self.teacher is self.student:
            return

        momentum = float(self.ema_coefficient)
        if momentum == 1.0:
            return

        student_params = list(self.student.parameters())
        teacher_params = list(self.teacher.parameters())
        if len(student_params) != len(teacher_params):
            raise RuntimeError("Student and teacher parameter counts differ")

        for student_param, teacher_param in zip(student_params, teacher_params):
            teacher_param.lerp_(student_param, 1.0 - momentum)
