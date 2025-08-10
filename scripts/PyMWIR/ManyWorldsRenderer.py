import torch
from . import PyBindMWIR
from . import Scene
from . import ManyWorlds



class ManyWorldsRendererFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, occupancy, normals, payload):
        inverse_renderer, scene, many_worlds, output, seed = payload

        result = inverse_renderer.Forward(scene.scene, many_worlds.many_worlds, output, seed)

        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result = ctx.saved_tensors

        print(grad_output)
        return None, None, None 





class ManyWorldsRenderer:
    def __init__(self):
        self.render_func = ManyWorldsRendererFunction.apply
        self.inverse_renderer = PyBindMWIR.ManyWorldsRenderer()

    def Render(self, scene, many_worlds, output=None, seed=None):
        if self.inverse_renderer is None:
            raise ValueError("InverseRenderer ownership has been transferred")
        if type(scene) is not Scene:
            raise TypeError("scene must be of type Scene")
        if type(many_worlds) is not ManyWorlds.ManyWorlds:    #Why not just ManyWorlds??????????????
            raise TypeError("many_worlds must be of type ManyWorlds")
        if output is not None and type(output) is not torch.Tensor:
            raise TypeError("output must be a torch.Tensor or None")

        occupancy = many_worlds.GetOccupancy()
        normal = many_worlds.GetNormal()
        return self.render_func(occupancy, normal, (self.inverse_renderer, scene, many_worlds, output, seed))