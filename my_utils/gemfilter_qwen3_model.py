# my_qwen_model.py
import os
import torch
import copy
import time
import gc
import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import sys
from my_utils.my_generation import set_select_mode, set_select_layer, reduce_layer, get_layer_context, recover_layer, my_greedy_generate

sys.path.append('/work/lei/GemFilter')
from my_baseline.GemFilter.qwen3_select_attention import Qwen3SelectAttention
from my_utils.utils import save_dict_incrementally_to_df

# def register_hooks_on_all_layers(model):
#     """
#     Register hooks on all layers to capture attention weights.
#     """

#     for i, layer in enumerate(model.model.layers):
#         # Register the hook for the attention module in each layer
#         attention_module = layer.self_attn
#         attention_module.layer_idx = i  # Store the layer index in the attention module for identification
#         attention_module.register_forward_hook(capture_all_layers_attn_weights_hook)


class MyQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, use_custom_generation=False, select_layer_idx=None, **kwargs):
        super().__init__(config)
        if 'topk' in kwargs:
            print("✅ kwargs has topk!!!!! Setting it in config.")
            setattr(config, 'topk', kwargs['topk'])
        self.use_custom_generation = use_custom_generation
        self.select_layer_idx = select_layer_idx
        # print("inspection of the model:", {self})
        
        modeling_qwen3.Qwen3Attention = Qwen3SelectAttention
        self.save_output_path = kwargs.get('output_path', None)

        # print(f"config within MyQwen3ForCausalLM: {config}")
        # print(f"kwargs within MyQwen3ForCausalLM: {kwargs}")
        self.output_attentions = kwargs.get('output_attentions', False)
        self.filename = "/collected_outputs.pkl"
        os.makedirs(self.save_output_path, exist_ok=True)
        print("output path:", self.save_output_path)
        self.i = 0


    @torch.no_grad()
    def make_attn_hook(self, layer_idx, pass_type):
        def hook_fn(module, input, output):
            
            attn_output = module.attn_output.detach().cpu()
            if pass_type == "first":
                self.first_pass_attn_outputs[layer_idx] = attn_output
            elif pass_type == "second":
                self.second_pass_attn_outputs[layer_idx] = attn_output
        return hook_fn

    
    @torch.no_grad()
    def my_greedy_generate_selection(self, input_ids, attention_mask=None, tokenizer=None, max_gen_len=50, select_layer_idx=None, print_context=False):
        new_entry = {}
        new_entry['sample_id'] = self.i
        new_entry['input_ids'] = input_ids
        new_entry['select_layer_idx'] = select_layer_idx
        overall_start = time.time()
        select_layer_idx = select_layer_idx if select_layer_idx is not None else self.select_layer_idx

        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        ### Timing: select mode setup ###
        t0 = time.time()
        set_select_mode(self, True)
        select_layer_idx = set_select_layer(self, select_layer_idx)
        # print(f"[Timing] Select mode setup: {time.time() - t0:.4f} s")

        ### Timing: layer reduction ###
        t0 = time.time()
        self, original_layers = reduce_layer(self, select_layer_idx)
        # print(f"[Timing] Layer reduction: {time.time() - t0:.4f} s")

        ### Timing: first forward pass ###
       
        t0 = time.time()

        # switch to hook, register hook before first pass
        # self.first_pass_attn_outputs = {}  # Clear previous
        # for layer_idx in range(select_layer_idx + 1):
        #     self.model.layers[layer_idx].self_attn._first_hook = \
        #         self.model.layers[layer_idx].self_attn.register_forward_hook(self.make_attn_hook(layer_idx, "first"))



        first_outputs = self(input_ids, attention_mask=attention_mask, output_attentions = self.output_attentions, output_hidden_states = True)
        # print(f"[Timing] First forward pass: {time.time() - t0:.4f} s")
        new_entry["first_pass_time"] = time.time() - t0
        
        
        # remove first hook
        for layer_idx in range(select_layer_idx + 1):
            self.model.layers[layer_idx].self_attn._first_hook.remove()

        
        # new_entry["first_pass_attention_output"] = {}
        # if self.output_attentions:
        #     for layer_idx in range(select_layer_idx + 1):
        #         # print("layer_idx:", layer_idx, "attention shape:",self.model.layers[layer_idx].self_attn.attn_output.cpu().numpy().shape)
        #         new_entry["first_pass_attention_output"][layer_idx] = self.model.layers[layer_idx].self_attn.attn_output.detach().cpu()

        
        new_entry[f"layer_{select_layer_idx}_per_head_token_indecies"] = self.model.layers[select_layer_idx].self_attn.per_head_token_indecies

        ### Timing: context extraction ###
        t0 = time.time()
        new_input_ids = get_layer_context(self, tokenizer, input_ids[0], select_layer_idx, print_context=print_context)
        # print(f"[Timing] Context extraction: {time.time() - t0:.4f} s")
        new_entry['selected_input_ids'] = new_input_ids

        ### Timing: layer recovery ###
        t0 = time.time()
        self = recover_layer(self, original_layers)
        set_select_mode(self, False)
        # print(f"[Timing] Layer recovery: {time.time() - t0:.4f} s")

        # register second hook, for second pass
        # self.second_pass_attn_outputs = {}  # Clear previous
        # for layer_idx in range(len(self.model.layers)):
        #     self.model.layers[layer_idx].self_attn._second_hook = \
        #         self.model.layers[layer_idx].self_attn.register_forward_hook(self.make_attn_hook(layer_idx, "second"))
             
        ### Timing: second forward pass ###
        t0 = time.time()
        outputs = self(new_input_ids, attention_mask=attention_mask, output_attentions = self.output_attentions, output_hidden_states = True) # return attention outputs and attention weights
        # print(f"[Timing] Second forward pass: {time.time() - t0:.4f} s")
        new_entry["second_pass_time"] = time.time() - t0
        
        # remove second hook
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].self_attn._second_hook.remove()

        # new_entry["second_pass_attention_output"] = {}
        # if self.output_attentions:
        #     for layer_idx in range(len(self.model.layers)):
        #         new_entry["second_pass_attention_output"][layer_idx] = self.model.layers[layer_idx].self_attn.attn_output.detach().cpu()
        
        # output: odict_keys(['logits', 'past_key_values', 'hidden_states', 'attentions']), attention laways return zero
       
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        ### Timing: generation loop ###
        t0 = time.time()
        output_ids = my_greedy_generate(self, tokenizer, pred_token_idx, past_key_values, max_gen_len=max_gen_len)
        # print(f"[Timing] Generation loop: {time.time() - t0:.4f} s")

        new_entry['output_ids'] = output_ids
        ### Timing: decoding ###
        t0 = time.time()
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        # print(f"[Timing] Decoding: {time.time() - t0:.4f} s")
        new_entry['response'] = response

        # print(f"[Overall timing] my_greedy_generate_selection total: {time.time() - overall_start:.4f} s")
        # print(f"Response: {response}")
        

        # Return as tensor for consistency with other models, add batch dimension
        output_ids = torch.tensor(output_ids, device=device).unsqueeze(0)
        full_output_ids = torch.cat([input_ids, output_ids], dim=-1) 

        # save the new entry to a file
        save_dict_incrementally_to_df(file_path = self.save_output_path + "/" +  self.filename, new_entry=new_entry)
        # torch.save(self.first_pass_attn_outputs, os.path.join(self.save_output_path, f"attn_first_pass_sample_{self.i}.pt"))
        # torch.save(self.second_pass_attn_outputs, os.path.join(self.save_output_path, f"attn_second_pass_sample_{self.i}.pt"))
        self.i += 1

        
        # clean up
        del new_entry  # Delete the tensor
        torch.cuda.empty_cache()  # Release the memory
        gc.collect()

        return full_output_ids

    def generate(self, input_ids, attention_mask=None, **kwargs):
        # print("here goes use custom generation")
        # print("use_custom_generation:", self.use_custom_generation)
        if self.use_custom_generation:
            max_gen_len = kwargs.get('max_new_tokens', None)
            # print("what is max_gen_len?", max_gen_len)
            results = self.my_greedy_generate_selection(input_ids, attention_mask, self.tokenizer, max_gen_len=max_gen_len)
            # print("results:", results)
            return results
        else:
            return super().generate(input_ids, attention_mask=attention_mask, **kwargs)
