import torch
from . import PyBindMWIR
from . import Scene

class Renderer:
    def __init__(self):
        self.renderer = PyBindMWIR.Renderer()

    def Render(self, scene, output=None, seed=None):
        if self.renderer is None:
            raise ValueError("Renderer ownership has been transferred")
        if type(scene) is not Scene:
            raise TypeError("scene must be of type Scene")
        if output is not None and type(output) is not torch.Tensor:
            raise TypeError("output must be a torch.Tensor or None")
        return self.renderer.Render(scene.scene, output, seed)