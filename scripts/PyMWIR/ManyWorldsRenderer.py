import torch
from . import PyBindMWIR
from . import Scene
from . import ManyWorlds



class ManyWorldsRendererFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, occupancy, payload):
        inverse_renderer, scene, many_worlds, output, seed = payload
        result = inverse_renderer.Forward(scene.scene, many_worlds.many_worlds, output, seed)
        ctx.inverse_renderer = inverse_renderer
        ctx.scene = scene
        ctx.many_worlds = many_worlds
        ctx.seed = seed
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inverse_renderer = ctx.inverse_renderer
        scene = ctx.scene
        many_worlds = ctx.many_worlds
        seed = ctx.seed
        occupancy_gradient = inverse_renderer.Backward(scene.scene, many_worlds.many_worlds, grad_output, None, seed)
        return occupancy_gradient, None 



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
        return self.render_func(occupancy, (self.inverse_renderer, scene, many_worlds, output, seed))