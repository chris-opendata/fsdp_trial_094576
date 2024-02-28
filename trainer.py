from __future__ import unicode_literals, print_function, division
import torch
from earlystop_cond import EarlyStopConditionByCount

class Trainer():
    def train_epoch(
        self,
        iepoch,
        args,
        train_dataloader,
        model,
        optimizer,
        lr_scheduler,
        accelerator,
        max_steps_stop,
        log_freq,
        logger
    ):
        '''
            train epoch
        '''
        early_stop = False
        n_train_iterations = len(train_dataloader)
        model.train()
        for ibatch, batch in enumerate(train_dataloader):
            outputs = model(batch)

            loss = outputs["cost"]
            loss_data = loss.item()

            if (ibatch+1) % log_freq == 0:
                logger.info((f'Train (Epoch {iepoch}, iBatch {ibatch}): Cost {loss_data}'))

            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)

            if (ibatch+1) % args.gradient_accumulation_steps == 0 or \
                ibatch == n_train_iterations - 1:
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad(set_to_none=True)
                max_steps_stop.incr()

            early_stop = max_steps_stop()
            if early_stop:
                break

        if (ibatch+1) % log_freq != 0: # Log remaning batch results
            logger.info((f'Train (Epoch {iepoch}, iBatch {ibatch}): Cost {loss_data}'))

        return early_stop

    def __call__(
        self,
        args,
        train_dataloader,
        model,
        optimizer,
        lr_scheduler,
        accelerator,
        logger
    ):
        max_steps_stop = EarlyStopConditionByCount(args.max_train_steps)
        for iepoch in range(args.num_train_epochs):
            logger.info(f'Epoch {iepoch}')

            torch.cuda.empty_cache()
      
            early_stop = self.train_epoch(iepoch=iepoch,
                                args=args,
                                train_dataloader=train_dataloader,
                                model=model,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                accelerator=accelerator,
                                max_steps_stop=max_steps_stop,
                                log_freq=args.log_freq,
                                logger=logger)
            accelerator.wait_for_everyone()

            if early_stop:
                break

        logger.info(f"Training completes @ epoch {iepoch}.")
