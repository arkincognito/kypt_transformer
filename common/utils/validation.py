from common.utils.summary_writer import write_summary
import torch


def validate(model, data_loader, val_writer, epoch, itr, trainer, cfg):
    model.eval()

    data_cnt = 0
    val_loss = dict()
    with torch.no_grad():
        for (inputs, targets, meta_info) in data_loader:
            model_out = model(inputs, targets, meta_info, "train")
            out = {k[:-4]: model_out[k] for k in model_out.keys() if "_out" in k}
            loss = {k: model_out[k] for k in model_out.keys() if "_out" not in k}
            loss = {k: loss[k].sum() for k in loss}

            loss["joint_heatmap"] *= cfg.hm_weight
            if cfg.has_object:
                loss["obj_seg"] *= cfg.obj_hm_weight

                loss["obj_rot"] *= cfg.obj_rot_weight
                loss["obj_trans"] *= cfg.obj_trans_weight
                loss["obj_corners"] *= cfg.obj_corner_weight
                loss["obj_corners_proj"] *= cfg.obj_corner_proj_weight
                loss["obj_weak_proj"] *= cfg.obj_weak_proj_weight

            if cfg.hand_type == "both":
                loss["rel_trans"] *= cfg.rel_trans_weight
                loss["hand_type"] *= cfg.hand_type_weight

            if cfg.predict_type == "angles":
                loss["pose"] *= cfg.pose_weight
                loss["shape_reg"] *= cfg.shape_reg_weight
                loss["vertex_loss"] *= cfg.vertex_weight
                loss["shape_loss"] *= cfg.shape_weight
                loss["joints_loss"] *= cfg.joint_weight
                loss["joints2d_loss"] *= cfg.joint_2d_weight

            elif cfg.predict_type == "vectors":
                if cfg.predict_2p5d:
                    loss["joint_2p5d_hm"] *= cfg.joint_2p5d_weight
                else:
                    loss["joint_vec"] *= cfg.joint_vec_weight
                    loss["joints_loss"] *= cfg.joint_weight
                    loss["joints2d_loss"] *= cfg.joint_2d_weight

            loss["cls"] *= cfg.cls_weight

            if cfg.use_2D_loss and ((not cfg.predict_2p5d) or (cfg.predict_type == "angles")):
                loss["cam_trans"] *= cfg.cam_trans_weight
                loss["cam_scale"] *= cfg.cam_scale_weight

            data_cnt += len(inputs["img"])

            for key in loss.keys():
                if key not in val_loss:
                    val_loss[key] = 0
                val_loss[key] += loss[key]

    for key in val_loss.keys():
        val_loss[key] /= data_cnt

    write_summary(val_writer, inputs, out, val_loss, epoch, itr, trainer, cfg)

    model.train()

    return val_loss
