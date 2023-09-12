import sys
import torch
from collections import OrderedDict
import ast


def main():

    # Optimize the number of parameters of a checkpoint

    if len(sys.argv) != 3:
        print('Error: Two input arguments, checkpoint_path and searched MSA architecture in a list.')
        return

    checkpoint_path = sys.argv[1]
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model']

    try:
        # Use ast.literal_eval to safely parse the string as a list
        MSA_indicators = ast.literal_eval(sys.argv[2])

        if not isinstance(MSA_indicators, list):
            raise ValueError("The provided parameter is not a valid list.")

        # Now you have the list parameter
        print(f"List Parameter: {MSA_indicators}")

    except (ValueError, SyntaxError) as e:
        print(f"Invalid MSA indicators: {e}")

    new_dict = OrderedDict()

    if any('bconv' in key for key in list(state_dict.keys())):
        print('Error: The checkpoint is already optimized!')
        return

    for k, v in state_dict.items():
        if 'head_probs' in k:
            block_name = k.replace('head_probs', '')
            block_num = int(k.split('.')[1])
            head_probs = (state_dict[k] / 1e-2).softmax(0)
            num_heads = head_probs.shape[0]
            feature_dim = state_dict[block_name + 'v.weight'].shape[0]
            head_dim = feature_dim // num_heads

            if MSA_indicators[block_num][-1] == 1:
                print('Error: checkpoint and MSA indicators do not match!')
                return

            new_v_weight = state_dict[block_name + 'v.weight'].view(num_heads, head_dim, feature_dim).permute(1, 2, 0) @ head_probs
            new_v_bias = state_dict[block_name + 'v.bias'].view(num_heads, head_dim).permute(1, 0) @ head_probs
            new_proj_weight = state_dict[block_name + 'proj.weight'].view(feature_dim, num_heads, head_dim).permute(0, 2, 1) @ head_probs

            if MSA_indicators[block_num][1] == 1:
                bn_name = 'bn_3x3.'
                new_dict[block_name + 'bconv.0.weight'] = new_v_weight.permute(2, 0, 1).view(3, 3, head_dim, -1).permute(2, 3, 1, 0)
                new_dict[block_name + 'bconv.0.bias'] = new_v_bias.sum(-1)
                new_dict[block_name + 'bconv.3.weight'] = new_proj_weight.sum(-1)[..., None, None]
            else:
                bn_name = 'bn_1x1.'
                new_dict[block_name + 'bconv.0.weight'] = new_v_weight[..., 4][..., None, None]
                new_dict[block_name + 'bconv.0.bias'] = new_v_bias[..., 4]
                new_dict[block_name + 'bconv.3.weight'] = new_proj_weight[..., 4][..., None, None]

            new_dict[block_name + 'bconv.3.bias'] = state_dict[block_name + 'proj.bias']

            new_dict[block_name + 'bconv.1.weight'] = state_dict[block_name + bn_name + 'weight']
            new_dict[block_name + 'bconv.1.bias'] = state_dict[block_name + bn_name + 'bias']
            new_dict[block_name + 'bconv.1.running_mean'] = state_dict[block_name + bn_name + 'running_mean']
            new_dict[block_name + 'bconv.1.running_var'] = state_dict[block_name + bn_name + 'running_var']
            new_dict[block_name + 'bconv.1.num_batches_tracked'] = state_dict[block_name + bn_name + 'num_batches_tracked']

        else:
            if len(k.split('.')) <= 4 or '.'.join(k.split('.')[:-2]) + '.head_probs' not in state_dict.keys():
                new_dict[k] = state_dict[k]

    torch.save({'model': new_dict}, '.'.join(checkpoint_path.split('.')[:-1]) + '_optimized.pth')


if __name__ == '__main__':
    main()
