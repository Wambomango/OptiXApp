import torch
from . import PyBindMWIR
from . import Mesh
from . import Antenna
from . import Signal
from . import Scene

class ForwardRenderer:
    def __init__(self, scene = None):
        scene = self.__SetScene(scene)
        self.forward_renderer = PyBindMWIR.ForwardRenderer(scene)

    def Render(self):
        if self.forward_renderer is None:
            raise ValueError("internal renderer instance has been moved.")
        return self.forward_renderer.Render()

    def SetScene(self, scene):
        if self.forward_renderer is None:
            raise ValueError("internal renderer instance has been moved.")
        scene = self.__SetScene(scene)
        self.forward_renderer.SetScene(scene)

    def GetScene(self):
        if self.forward_renderer is None:
            raise ValueError("internal renderer instance has been moved.")
        return Scene(None, None, None, None, self.forward_renderer.GetScene())

    def __SetScene(self, scene):
        if scene is None:
            return PyBindMWIR.Scene()
        if type(scene) is not Scene:
            raise TypeError("scene must be of type Scene")
        tmp = scene.scene
        scene.scene = None
        return tmp