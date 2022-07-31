import torchvision


def write_summary(writer, inputs, out, loss, epoch, itr, trainer, cfg):
    # dump the outputs
    img_grid = torchvision.utils.make_grid(inputs["img"][:4])
    writer.add_image("input patches", img_grid, epoch * len(trainer.batch_generator) + itr)
    hm_grid = torchvision.utils.make_grid(out["joint_heatmap"][:4].unsqueeze(1), normalize=True)

    if cfg.has_object:
        seg_grid_gt = torchvision.utils.make_grid(out["obj_kps_gt"][:4].unsqueeze(1), normalize=True)
        seg_grid_pred = torchvision.utils.make_grid(out["obj_seg_pred"][:4].unsqueeze(1), normalize=True)
        writer.add_image("seg gt patches", seg_grid_gt, epoch * len(trainer.batch_generator) + itr)
        writer.add_image("seg pred patches", seg_grid_pred, epoch * len(trainer.batch_generator) + itr)

    writer.add_image("heatmap", hm_grid, epoch * len(trainer.batch_generator) + itr)

    # dump the losses
    if cfg.predict_type == "angles":
        writer.add_scalar("pose loss", loss["pose"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("shape_reg loss", loss["shape_reg"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("vertex loss", loss["vertex_loss"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("shape", loss["shape_loss"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("joints loss", loss["joints_loss"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("joints2d_loss loss", loss["joints2d_loss"], epoch * len(trainer.batch_generator) + itr)

    elif cfg.predict_type == "vectors":
        if cfg.predict_2p5d:
            writer.add_scalar("joint_2p5d_hm loss", loss["joint_2p5d_hm"], epoch * len(trainer.batch_generator) + itr)
        else:
            writer.add_scalar("joint_vec loss", loss["joint_vec"], epoch * len(trainer.batch_generator) + itr)
            writer.add_scalar("joints loss", loss["joints_loss"], epoch * len(trainer.batch_generator) + itr)
            writer.add_scalar("joints2d_loss loss", loss["joints2d_loss"], epoch * len(trainer.batch_generator) + itr)

    if cfg.hand_type == "both":
        writer.add_scalar("rel_trans loss", loss["rel_trans"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("hand_type loss", loss["hand_type"], epoch * len(trainer.batch_generator) + itr)

    if cfg.has_object:
        writer.add_scalar("obj_seg loss", loss["obj_seg"], epoch * len(trainer.batch_generator) + itr)

        writer.add_scalar("obj_rot loss", loss["obj_rot"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("obj_trans loss", loss["obj_trans"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("obj_corners loss", loss["obj_corners"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("obj_corners_proj loss", loss["obj_corners_proj"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("obj_weak_proj loss", loss["obj_weak_proj"], epoch * len(trainer.batch_generator) + itr)

    writer.add_scalar("cls loss", loss["cls"], epoch * len(trainer.batch_generator) + itr)

    writer.add_scalar("heatmap loss", loss["joint_heatmap"], epoch * len(trainer.batch_generator) + itr)

    writer.add_scalar(
        "Learning rate", trainer.lr_scheduler.get_last_lr()[-1], epoch * len(trainer.batch_generator) + itr
    )
    writer.add_scalar("total loss", sum(loss[k] for k in loss), epoch * len(trainer.batch_generator) + itr)

    if cfg.use_2D_loss and ((not cfg.predict_2p5d) or (cfg.predict_type == "angles")):
        writer.add_scalar("cam trans", loss["cam_trans"], epoch * len(trainer.batch_generator) + itr)
        writer.add_scalar("cam scale", loss["cam_scale"], epoch * len(trainer.batch_generator) + itr)
