import os
import yaml

i = 501
for backbone_type in ['resnet']:
    for sim_coeff in [25.0 / 2, 25.0, 50.0]:
        for std_coeff in [25.0 / 2, 25.0, 50.0]:
            for start_lr in [0.15, 0.3, 0.6, 1.2, 2.4]:
                for planes_multipliers in [
                    [2, 2, 2],
                    [2, 4, 2],
                    [2, 8, 2],
                ]:
                    comment = f"RandomMask_{i}_backbone_type_{backbone_type}_planes_multipliers_{planes_multipliers}_sim_coeff_{sim_coeff}_std_coeff_{std_coeff}_lr_{start_lr}"
                    comment = comment.replace(" ", "").replace("[", "").replace("]", "").replace(",", "_").replace("'", "")
                    config = {
                        "comment": comment,
                        "backbone_type": backbone_type,
                        "sim_coeff": sim_coeff,
                        "std_coeff": std_coeff,
                        "start_lr": start_lr,
                        "planes_multipliers": planes_multipliers,
                    }
                    with open(os.path.join("configs", f"experiment_{i}.yml"), 'w') as conf_file:
                        yaml.dump(config, conf_file)
                
                    i += 1
