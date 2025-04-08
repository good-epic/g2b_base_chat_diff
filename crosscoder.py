# %%
from dictionary_learning import CrossCoder
from nnsight import LanguageModel
import torch as th
# %%

crosscoder = CrossCoder.from_pretrained("Butanium/gemma-2-2b-crosscoder-l13-mu4.1e-02-lr1e-04", from_hub=True)
gemma_2 = LanguageModel("google/gemma-2-2b", device_map="cuda")
gemma_2_it = LanguageModel("google/gemma-2-2b-it", device_map="cuda")
prompt = "quick fox brown"

with gemma_2.trace(prompt):
    l13_act_base = gemma_2.model.layers[13].output[0][:, -1].save() # (1, 2304)
    gemma_2.model.layers[13].output.stop()

with gemma_2_it.trace(prompt):
    l13_act_it = gemma_2_it.model.layers[13].output[0][:, -1].save() # (1, 2304)
    gemma_2_it.model.layers[13].output.stop()


crosscoder_input = th.cat([l13_act_base, l13_act_it], dim=0).unsqueeze(0).cpu() # (batch, 2, 2304)
print(crosscoder_input.shape)
reconstruction, features = crosscoder(crosscoder_input, output_features=True)

# print metrics
print(f"MSE loss: {th.nn.functional.mse_loss(reconstruction, crosscoder_input).item():.2f}")
print(f"L1 sparsity: {features.abs().sum():.1f}")
print(f"L0 sparsity: {(features > 1e-4).sum()}")
# %%
