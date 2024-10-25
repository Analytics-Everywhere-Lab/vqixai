
def cam_explainer(xai_method, model, x_tensor, raw_img, target_layer, targets):
    with xai_method(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=x_tensor, targets=targets)[0, :]
        explanation_map = show_cam_on_image(raw_img, grayscale_cam, use_rgb=True)
        return explanation_map
