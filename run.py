import time

import pytorch_lightning as pl
from utils.ConfigData import NetConfigData, SuperParams
from utils.generalUtils import (
    getDataModule,
    getModel,
    getTrainer,
)
from utils.tools import findLR


def runOneNet(netID):
    cfg = NetConfigData.getNetConfig(netID)
    print(
        "Start ...",
        netID,
        "time:",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)

    # print("Start preProcessData...")
    # dataPathConfig = preProcessData()
    # data_module = getDataModule(dataPathConfig=dataPathConfig, cfg=cfg)

    data_module = getDataModule(cfg=cfg)
    regression_model = getModel(cfg=cfg)
    trainer = getTrainer(cfg=cfg)

    print("start fitting")
    # Launch training session

    if cfg.runConfig.test:
        print("start testing")
        trainer.test(
            regression_model,
            datamodule=data_module,
            ckpt_path=cfg.runConfig.testCheckpointFilePath,
        )
    else:
        if cfg.runConfig.resumeTrainFromCheckpoint:
            assert cfg.runConfig.resumeCheckpointFilePath is not None, (
                "resumeCheckpointFilePath is None"
            )
            trainer.fit(
                regression_model,
                datamodule=data_module,
                ckpt_path=cfg.runConfig.resumeCheckpointFilePath,
            )
        else:
            if cfg.runConfig.findLr:
                lr = findLR(trainer, data_module, regression_model)
                regression_model.lr = lr
            trainer.fit(regression_model, data_module)

        trainer.test(regression_model, datamodule=data_module)

    print(
        "End ...",
        netID,
        "time:",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )


def run():
    print(
        "Start running..., time is:",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )

    for runNetID in SuperParams.runNets:
        runOneNet(runNetID)

    print(
        "run finished, time is:",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )


# Main execution block
if __name__ == "__main__":
    run()
