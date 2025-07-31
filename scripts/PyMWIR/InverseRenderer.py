import torch
from . import PyBindMWIR
from . import Mesh
from . import Antenna
from . import Signal
from . import Scene

class InverseRenderer:
    def __init__(self):
        self.inverse_renderer = PyBindMWIR.InverseRenderer()

    def Render(self, scene, output=None):
        if self.inverse_renderer is None:
            raise ValueError("InverseRenderer ownership has been transferred")
        if type(scene) is not Scene:
            raise TypeError("scene must be of type Scene")
        if output is not None and type(output) is not torch.Tensor:
            raise TypeError("output must be a torch.Tensor or None")
        return self.inverse_renderer.Render(scene.scene, output)
