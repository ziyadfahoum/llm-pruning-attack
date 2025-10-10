from llmcompressor import Oneshot
from llmcompressor.datasets import get_calibration_dataloader
from llmcompressor.logger import logger


class OneShotWithoutSave(Oneshot):
    def __call__(self):
        """
        Run the one-shot compression process without saving the compressed model.
        Suppresses all logging output during execution for cleaner output.
        """
        logger.disable("llmcompressor")
        logger.info("Starting one-shot pruning.")
        try:
            calibration_dataloader = get_calibration_dataloader(self.dataset_args, self.processor)
            self.apply_recipe_modifiers(
                calibration_dataloader=calibration_dataloader,
                recipe_stage=self.recipe_args.stage,
            )
            # this part is not necessary for our purpose
            # post_process(
            #     model_args=self.model_args,
            #     recipe_args=self.recipe_args,
            #     output_dir=self.output_dir,
            # )
        finally:
            logger.info("Finished one-shot pruning.")
            logger.enable("llmcompressor")
