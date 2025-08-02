import torch
from . import PyBindMWIR
from . import Scene

class ForwardRenderer:
    def __init__(self):
        self.forward_renderer = PyBindMWIR.ForwardRenderer()

    def Render(self, scene, output=None):
        if self.forward_renderer is None:
            raise ValueError("InverseRenderer ownership has been transferred")
        if type(scene) is not Scene:
            raise TypeError("scene must be of type Scene")
        if output is not None and type(output) is not torch.Tensor:
            raise TypeError("output must be a torch.Tensor or None")
        return self.forward_renderer.Render(scene.scene, output)