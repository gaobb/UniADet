source "$(dirname "$0")/env.sh"

train_dataset=visa
test_dataset=Real-IAD-Variety


for pretrained_model in 'ViT-L-14@336px' 'ViT-L-14-DINOv2Reg' 'ViT-L-16-DINOv3' 
do
    case "$pretrained_model" in
    "ViT-L-14@336px"|"ViT-L-14-DINOv2Reg")
        image_size=518
        ;;
    "ViT-L-16-DINOv3")
        image_size=512
        ;;
    *)
        image_size=518
        ;;
    esac
    echo $image_size

    save_path=${UNIADET_RESULTS}/test_${test_dataset}_${train_dataset}_${pretrained_model}
    checkpoint_path=${UNIADET_CKPT}/train_on_${train_dataset}_zs_${pretrained_model}_imagesize${image_size}_1015
    epoch=15
    
    # for zero-shot
    lamda_pixel=0.5 # 0.1 for zero-shot and 0.5 for few-shot
    lamda_fs=0.8
    k_shot=0
    
    for obj_list in "2pin_block_plug"   "3_adapter"   "3pin_aviation_connector"   "4_wire_stepping_motor"   "D-sub-connector"   "DVD_switch"   "LED_indicator"  \
                "PLCC_socket"   "VR_joystick"   "access_card"   "accurate_detection_switch"   "aircraft_model_head"   "angled_toggle_switch"   "audio_jack_socket" \
                "bag_buckle"   "ball_pin"   "balun_transformer"   "battery"   "battery_holder_connector"   "battery_socket_connector"   "bend_connector"   "blade_switch" \
                "blue_light_switch"   "bluetooth_module"   "boost_converter_module"   "bread_model"   "brooch_clasp_accessory"   "button_battery_holder"   "button_motor" \
                "button_switch"   "car_door_lock_switch"   "ceramic_fuse"   "ceramic_wave_filter"   "charging_port"   "chip-inductor"   "circuit_breaker" \
                "circular_aviation_connector"   "common-mode-choke"   "common-mode-filter"   "connector"   "connector_housing-female"   "console_switch" \
                "crimp_st_cable_mount_box"   "dc_jack"   "dc_power_connector"   "detection_switch"   "duckbill_circuit_breaker"   "earphone_audio_unit" \
                "effect_transistor"   "electronic_watch_movement"   "ethernet-connector"   "ferrite_bead"   "ffc_connector_plug"   "flow_control_valve" \
                "flower_copper_shape"   "flower_velvet_fabric"   "fork-crimp-terminal"   "fuse_cover"   "fuse_holder"   "gear"   "gear_motor"   "green-ring-filter" \
                "hairdryer_switch"   "hall_effect_sensor"   "headphone_jack-female"   "headphone_jack_socket"   "hex_plug"   "humidity_sensor"   "ingot_buckle" \
                "insect_metal_parts"   "inverter_connector"   "jam_jar_model"   "joystick_switch"   "kfc_push-key_switch"   "knob-cap"   "laser_diode"   "lattice_block_plug"  \
                "lego-pin-connector-plate"   "lego-propeller"   "lego-reel"   "lego-technical-gear"   "lego-turbine"   "lighting_connector"   "lilypad_led"   "limit-switch" \
                "lithium_battery_plug"   "littel-fuse"   "little_cow_model"   "lock"   "long-zipper"   "meteor_hammer_arrowhead"   "miniature_laser_module" \
                "miniature_lifting_motor"   "miniature_motor"   "miniature_stepper_motor"   "mobile_charging_connector"   "model_steering_module"   "monitor_socket"  \
                "motor_bracket"   "motor_gear_reducer"   "motor_plug"   "mouse_socket"   "multi_function_switch"   "nylon_ball_head"   "optical_fiber_outlet" \
                "pencil_sharpener"   "pinboard_connector"   "pitch_connector"   "pneumatic_elbow"   "pot_core"   "potentiometer"   "power_bank_module" \
                "power_inductor"   "power_jack"   "power_strip_socket"   "pulse_transformer"   "purple-clay-pot"   "push_button_switch"   "push_in_terminal" \
                "recorder_switch"   "rectangular-connector-accessories"   "red_terminal"   "retaining_ring"   "rheostat"   "rotary_position_sensor" \
                "round_twist_switch"   "self-lock_switch"   "shelf_support_module"   "side_press_switch"   "side_release_buckle"   "silicon_cell_sensor"   "sim_card_reader" \
                "single-pole_potentiometer"   "single_switch"   "slide_switch"   "slipper- model"   "small_leaf"   "smd_receiver_module"   "solderless_adapter" \
                "spherical_airstone"   "spring_antenna"   "square_terminal"   "steering_T-head"   "steering_wheel"   "suction-cup"   "telephone_spring_switch" \
                "tension_snap_release_clip"   "thumbtack"   "thyristor"   "toy-tire"   "traceless_hair_clip"   "travel_switch"   "travel_switch_green" \
                "tubular_key_switch"   "vacuum_switch"   "vehicle_harness_conductor"   "vertical-adjustable-resistor"   "vibration_motor" \
                "volume_potentiometer"   "wireless_receiver_module"
    do
    
    CUDA_VISIBLE_DEVICES=${UNIADET_DEVICE} python test.py --dataset ${test_dataset} --test_data_path ${UNIADET_DATA}/${test_dataset} \
    --save_path ${save_path} \
    --pretrained_model ${pretrained_model} \
    --checkpoint_path ${checkpoint_path}/epoch_${epoch}.pth \
    --k_shot ${k_shot} \
    --window_list 1  \
    --lamda_fs $lamda_fs \
    --lamda_pixel $lamda_pixel \
    --features_list 12 15 18 21 24 --image_size $image_size --batch_size 32 \
    --class_name "${obj_list}"
    done  
    

    # for few-shot
    for k_shot in 1 2 4
    do
        for seed in 10 20 30
        do
            lamda_pixel=0.5 # 0.1 for zero-shot and 0.5 for few-shot
            lamda_fs=0.8
            for obj_list in "2pin_block_plug"   "3_adapter"   "3pin_aviation_connector"   "4_wire_stepping_motor"   "D-sub-connector"   "DVD_switch"   "LED_indicator"  \
                "PLCC_socket"   "VR_joystick"   "access_card"   "accurate_detection_switch"   "aircraft_model_head"   "angled_toggle_switch"   "audio_jack_socket" \
                "bag_buckle"   "ball_pin"   "balun_transformer"   "battery"   "battery_holder_connector"   "battery_socket_connector"   "bend_connector"   "blade_switch" \
                "blue_light_switch"   "bluetooth_module"   "boost_converter_module"   "bread_model"   "brooch_clasp_accessory"   "button_battery_holder"   "button_motor" \
                "button_switch"   "car_door_lock_switch"   "ceramic_fuse"   "ceramic_wave_filter"   "charging_port"   "chip-inductor"   "circuit_breaker" \
                "circular_aviation_connector"   "common-mode-choke"   "common-mode-filter"   "connector"   "connector_housing-female"   "console_switch" \
                "crimp_st_cable_mount_box"   "dc_jack"   "dc_power_connector"   "detection_switch"   "duckbill_circuit_breaker"   "earphone_audio_unit" \
                "effect_transistor"   "electronic_watch_movement"   "ethernet-connector"   "ferrite_bead"   "ffc_connector_plug"   "flow_control_valve" \
                "flower_copper_shape"   "flower_velvet_fabric"   "fork-crimp-terminal"   "fuse_cover"   "fuse_holder"   "gear"   "gear_motor"   "green-ring-filter" \
                "hairdryer_switch"   "hall_effect_sensor"   "headphone_jack-female"   "headphone_jack_socket"   "hex_plug"   "humidity_sensor"   "ingot_buckle" \
                "insect_metal_parts"   "inverter_connector"   "jam_jar_model"   "joystick_switch"   "kfc_push-key_switch"   "knob-cap"   "laser_diode"   "lattice_block_plug"  \
                "lego-pin-connector-plate"   "lego-propeller"   "lego-reel"   "lego-technical-gear"   "lego-turbine"   "lighting_connector"   "lilypad_led"   "limit-switch" \
                "lithium_battery_plug"   "littel-fuse"   "little_cow_model"   "lock"   "long-zipper"   "meteor_hammer_arrowhead"   "miniature_laser_module" \
                "miniature_lifting_motor"   "miniature_motor"   "miniature_stepper_motor"   "mobile_charging_connector"   "model_steering_module"   "monitor_socket"  \
                "motor_bracket"   "motor_gear_reducer"   "motor_plug"   "mouse_socket"   "multi_function_switch"   "nylon_ball_head"   "optical_fiber_outlet" \
                "pencil_sharpener"   "pinboard_connector"   "pitch_connector"   "pneumatic_elbow"   "pot_core"   "potentiometer"   "power_bank_module" \
                "power_inductor"   "power_jack"   "power_strip_socket"   "pulse_transformer"   "purple-clay-pot"   "push_button_switch"   "push_in_terminal" \
                "recorder_switch"   "rectangular-connector-accessories"   "red_terminal"   "retaining_ring"   "rheostat"   "rotary_position_sensor" \
                "round_twist_switch"   "self-lock_switch"   "shelf_support_module"   "side_press_switch"   "side_release_buckle"   "silicon_cell_sensor"   "sim_card_reader" \
                "single-pole_potentiometer"   "single_switch"   "slide_switch"   "slipper- model"   "small_leaf"   "smd_receiver_module"   "solderless_adapter" \
                "spherical_airstone"   "spring_antenna"   "square_terminal"   "steering_T-head"   "steering_wheel"   "suction-cup"   "telephone_spring_switch" \
                "tension_snap_release_clip"   "thumbtack"   "thyristor"   "toy-tire"   "traceless_hair_clip"   "travel_switch"   "travel_switch_green" \
                "tubular_key_switch"   "vacuum_switch"   "vehicle_harness_conductor"   "vertical-adjustable-resistor"   "vibration_motor" \
                "volume_potentiometer"   "wireless_receiver_module"
            do
                CUDA_VISIBLE_DEVICES=${UNIADET_DEVICE} python test.py --dataset ${test_dataset} --test_data_path ${UNIADET_DATA}/${test_dataset} \
                --save_path ${save_path} \
                --pretrained_model ${pretrained_model} \
                --checkpoint_path ${checkpoint_path}/epoch_${epoch}.pth \
                --seed ${seed} \
                --k_shot ${k_shot} \
                --window_list 1  \
                --lamda_fs $lamda_fs \
                --lamda_pixel $lamda_pixel \
                --features_list 12 15 18 21 24 --image_size $image_size --batch_size 32 \
                --class_name "${obj_list}"
            done
        done
    done
done
