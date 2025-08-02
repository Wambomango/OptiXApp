import torch
from . import PyBindMWIR
from . import Scene
from . import ManyWorlds

class InverseRenderer:
    def __init__(self):
        self.inverse_renderer = PyBindMWIR.InverseRenderer()

    def Render(self, scene, many_worlds, output=None):
        if self.inverse_renderer is None:
            raise ValueError("InverseRenderer ownership has been transferred")
        if type(scene) is not Scene:
            raise TypeError("scene must be of type Scene")
        if type(many_worlds) is not ManyWorlds.ManyWorlds:    #Why not just ManyWorlds??????????????
            raise TypeError("many_worlds must be of type ManyWorlds")
        if output is not None and type(output) is not torch.Tensor:
            raise TypeError("output must be a torch.Tensor or None")

        return self.inverse_renderer.Render(scene.scene, many_worlds.many_worlds, output)
