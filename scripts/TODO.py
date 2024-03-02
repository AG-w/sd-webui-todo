import math
import torch
import torch.nn.functional as F
import gradio as gr
from modules import scripts
from modules import shared
from typing import Type, Dict, Any, Tuple, Callable

def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method="nearest"):
    batch_size = item.shape[0]

    item = item.reshape(batch_size, cur_h, cur_w, -1).permute(0, 3, 1, 2)
    item = F.interpolate(item, size=(new_h, new_w), mode=method).permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item

def compute_merge(x: torch.Tensor, todo_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = todo_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = todo_info["args"]
    downsample_factor_1 = args['downsample_factor']
    downsample_factor_2 = args['downsample_factor_level_2']	
    downsample_factor_3 = args['downsample_factor_level_3']
	
    cur_h = original_h // downsample
    cur_w = original_w // downsample
    m = lambda v: v
    if downsample == 1 and downsample_factor_1 > 1:
        new_h = int(cur_h / downsample_factor_1)
        new_w = int(cur_w / downsample_factor_1)
        m = lambda v: up_or_downsample(v, cur_w, cur_h, new_w, new_h, args["downsample_method"])	
    elif downsample == 2 and downsample_factor_2 > 1:
        new_h = int(cur_h / downsample_factor_2)
        new_w = int(cur_w / downsample_factor_2)
        m = lambda v: up_or_downsample(v, cur_w, cur_h, new_w, new_h, args["downsample_method"])
    elif downsample == 4 and downsample_factor_3 > 1:
        new_h = int(cur_h / downsample_factor_3)
        new_w = int(cur_w / downsample_factor_3)
        m = lambda v: up_or_downsample(v, cur_w, cur_h, new_w, new_h, args["downsample_method"])
		
    return m

class ToDo(scripts.Script):
    #sorting_priority = 50
    #is_in_high_res_fix = False

    def title(self):
        return "Token Downsampling"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            todo_enabled = gr.Checkbox(label='Enabled', value=False)
            #todo_enabled_hr = gr.Checkbox(label='Enable only during hires fix', value=False)
            todo_downsample_method = gr.Dropdown(label="Downsample method", choices=["nearest", "bilinear", "bicubic", "nearest-exact"], value="nearest-exact")
            todo_downsample_factor_depth_1 = gr.Slider(label='Downsample Factor Depth 1', minimum=1.0, maximum=10.0, step=0.01, value=2.0)
            todo_downsample_factor_depth_2 = gr.Slider(label='Downsample Factor Depth 2', minimum=1.0, maximum=10.0, step=0.01, value=1.0)
        self.infotext_fields = (
            (todo_enabled, lambda d: gr.Checkbox.update(value="todo_enabled" in d)),
            (todo_downsample_method, "todo_downsample_method"),
            (todo_downsample_factor_depth_1, "todo_downsample_factor_depth_1"),
            (todo_downsample_factor_depth_2, "todo_downsample_factor_depth_2"),)
        return todo_enabled, todo_downsample_method, todo_downsample_factor_depth_1, todo_downsample_factor_depth_2

    #def before_hr(self, p, *script_args, **kwargs):
        #self.is_in_high_res_fix = True

    def process(self, p, *script_args, **kwargs):
        todo_enabled, todo_downsample_method, todo_downsample_factor_depth_1, todo_downsample_factor_depth_2 = script_args

        #if not p.enable_hr:
            #self.is_in_high_res_fix = False

        if not todo_enabled:
            return

        #if todo_enabled_hr and not self.is_in_high_res_fix:
            #return

        apply_patch(
            shared.sd_model, 
            downsample_factor = todo_downsample_factor_depth_1, 
            downsample_factor_level_2 = todo_downsample_factor_depth_2, 
            downsample_method = todo_downsample_method)


        p.extra_generation_params["todo_enabled"] = todo_enabled
        p.extra_generation_params["todo_downsample_method"] = todo_downsample_method
        p.extra_generation_params["todo_downsample_factor_depth_1"] = todo_downsample_factor_depth_1
        p.extra_generation_params["todo_downsample_factor_depth_2"] = todo_downsample_factor_depth_2

        #self.is_in_high_res_fix = False

        return
		
    def postprocess(self, p, processed, *args):
        todo_enabled, todo_downsample_method, todo_downsample_factor_depth_1, todo_downsample_factor_depth_2 = args
        #remove_patch(shared.sd_model)
        return

def make_todo_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToDoBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None, mask = None) -> torch.Tensor:
            c = context if self.disable_self_attn else None
            c = self.norm1(x) if c is None else c
            m = compute_merge(x, self._todo_info)

            x = self.attn1(self.norm1(x), context = m(c)) + x			
            x = self.attn2(self.norm2(x), context = context) + x
            x = self.ff(self.norm3(x)) + x
            return x
    
    return ToDoBlock

def hook_todo_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._todo_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None
    model._todo_info["hooks"].append(model.register_forward_pre_hook(hook))

def apply_patch(
        model: torch.nn.Module,
        downsample_method: str = "nearest-exact",
        downsample_factor: float = 2,
        downsample_factor_level_2: float = 1,
        downsample_factor_level_3: float = 1,
        ):

    # Make sure the module is not currently patched
    remove_patch(model)
    diffusion_model = model.model.diffusion_model
	
    diffusion_model._todo_info = {
        "size": None,
        "hooks": [],
        "args": {
            "downsample_method": downsample_method, # native torch interpolation methods ["nearest", "linear", "bilinear", "bicubic", "nearest-exact"]
            "downsample_factor": downsample_factor,  # amount to downsample by
            "downsample_factor_level_2": downsample_factor_level_2, # amount to downsample by at the 2nd down block of unet
            "downsample_factor_level_3": downsample_factor_level_3,
        }
    }
    hook_todo_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            make_todo_block_fn = make_todo_block
            module.__class__ = make_todo_block_fn(module.__class__)
            module._todo_info = diffusion_model._todo_info

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn"):
                module.disable_self_attn = False

    return model

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """

    for _, module in model.named_modules():
        if hasattr(module, "_todo_info"):
            for hook in module._todo_info["hooks"]:
                hook.remove()
            module._todo_info["hooks"].clear()

        if module.__class__.__name__ == "ToDoBlock":
            module.__class__ = module._parent
    
    return model

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False
