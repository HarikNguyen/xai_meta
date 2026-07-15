
def get_layer_parameters_map(baselearner, theta_0):
    """
    Map the flat theta_0 parameter list back onto the actual layers of the baselearner.
    """
    param_iterator = iter(theta_0)
    net_info_with_params = []

    # Iterate over sub-modules holding parameters, in PyTorch's own order
    for name, module in baselearner.named_modules():
        # Only consider modules with local parameters (excludes wrapping parent modules)
        local_params = list(module.parameters(recurse=False))
        if len(local_params) > 0:
            layer_dict = {
                "name": name,
                "type": module.__class__.__name__,
                "params": []
            }
            # Pull exactly as many tensors from theta_0 as this layer has parameters
            for _ in range(len(local_params)):
                layer_dict["params"].append(next(param_iterator))

            net_info_with_params.append(layer_dict)

    return net_info_with_params
