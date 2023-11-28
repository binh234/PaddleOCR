import glob
import os
import pickle
import time
from typing import Optional, Set, Tuple, Union

import paddle

class Checkpointer:
    """
    This class implements the functionality for checkpointing your model and training state
    during training.

    # Parameters

    keep_most_recent : `int`, optional (default=`2`)
        Sets the number of model checkpoints to keep on disk. If both `keep_most_recent` and
        `keep_most_recent_by_age` are set, we'll keep checkpoints that satisfy either criterion.
        If both are `None`, we keep all checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, os.PathLike],
        keep_most_recent: Optional[int] = 2,
        **kwargs
    ) -> None:
        self._checkpoint_dir = str(checkpoint_dir)
        self._best_checkpoint_dir = os.path.join(checkpoint_dir, "best_model")
        self._keep_most_recent = keep_most_recent
        self._last_save_time = time.time()
        self._rank = paddle.distributed.get_rank()

        if self._is_primary and not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)
            os.makedirs(self._best_checkpoint_dir)

    @property
    def _is_primary(self) -> bool:
        return self._rank == 0

    def _find_all_checkpoints(self) -> Set[Tuple[int, int]]:
        """Returns a set of integers, each of which is a number of batches that were completed at the
        time a checkpoint wsa saved."""
        checkpoints = set()
        pattern = "epoch*.pdparams"
        for model_state_file in glob.iglob(os.path.join(self._checkpoint_dir, pattern)):
            checkpoint_name = model_state_file.rsplit(".")[0]
            epoch = checkpoint_name.rsplit("_", 1)[-1]
            checkpoints.add((int(epoch), checkpoint_name))
        return checkpoints

    def _remove_checkpoint(self, checkpoint_name: str):
        param_path = f"{checkpoint_name}.pdparams"
        opt_path = f"{checkpoint_name}.pdopt"
        metric_path = f"{checkpoint_name}.states"
        student_path = f"{checkpoint_name}_student.pdparams"

        if os.path.exists(param_path):
            os.remove(param_path)
        if os.path.exists(opt_path):
            os.remove(opt_path)
        if os.path.exists(metric_path):
            os.remove(metric_path)
        if os.path.exists(student_path):
            os.remove(student_path)

    def _extract_student_weights(self, all_params, student_prefix="Student."):
        s_params = {
            key[len(student_prefix) :]: all_params[key]
            for key in all_params
            if student_prefix in key
        }
        return s_params

    def save_checkpoint(
        self,
        model,
        optimizer,
        logger,
        config,
        epoch=0,
        prefix=None,
        save_student_model=False,
        **kwargs,
    ) -> None:
        if not self._is_primary or self._checkpoint_dir is None:
            return

        kwargs.update({"epoch": epoch})
        checkpoint_name = prefix or f"epoch_{epoch}"
        model_path = os.path.join(self._checkpoint_dir, checkpoint_name)

        paddle.save(optimizer.state_dict(), model_path + '.pdopt')
        params_state_dict = model.state_dict()
        is_nlp_model = config['Architecture']["model_type"] == 'kie' and config[
        "Architecture"]["algorithm"] not in ["SDMGR"]
        if is_nlp_model is not True:
            paddle.save(params_state_dict, model_path + '.pdparams')
            metric_prefix = model_path

            if prefix == 'best_accuracy':
                paddle.save(params_state_dict,
                            os.path.join(self._best_checkpoint_dir, 'model.pdparams'))

        else:  # for kie system, we follow the save/load rules in NLP
            if config['Global']['distributed']:
                arch = model._layers
            else:
                arch = model
            if config["Architecture"]["algorithm"] in ["Distillation"]:
                arch = arch.Student
            arch.backbone.model.save_pretrained(model_path)
            metric_prefix = os.path.join(model_path, 'metric')

            if prefix == 'best_accuracy':
                arch.backbone.model.save_pretrained(self._best_checkpoint_dir)

        if save_student_model:
            s_params = self._extract_student_weights(params_state_dict)
            if len(s_params) > 0:
                paddle.save(s_params, model_path + "_student.pdparams")
                if prefix == 'best_accuracy':
                    paddle.save(params_state_dict,
                            os.path.join(self._best_checkpoint_dir, 'student.pdparams'))

        # save metric and config
        with open(metric_prefix + '.states', 'wb') as f:
            pickle.dump(kwargs, f, protocol=2)
        logger.info("Save model in {}".format(model_path))

        self._last_save_time = time.time()

        if self._is_primary and self._keep_most_recent is not None:
            checkpoints = list(self._find_all_checkpoints())
            checkpoints.sort(reverse=True)

            # Keep the most recent n checkpoints
            checkpoints_to_keep = set(checkpoints[: self._keep_most_recent])

            # Remove everything we're not keeping
            for checkpoint in checkpoints:
                if checkpoint not in checkpoints_to_keep:
                    self._remove_checkpoint(checkpoint[1])

    def find_latest_checkpoint(self) -> str:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        checkpoints = self._find_all_checkpoints()
        if len(checkpoints) <= 0:
            return None
        return checkpoints[-1][1]
