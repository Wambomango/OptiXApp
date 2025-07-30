import torch
from . import PyBindMWIR
from . import Mesh
from . import Antenna
from . import Signal
from . import Scene

class Renderer:
    def __init__(self, scene = None):
        scene = self.__SetScene(scene)
        self.renderer = PyBindMWIR.Renderer(scene)

    def Render(self):
        if self.renderer is None:
            raise ValueError("internal renderer instance has been moved.")
        return self.renderer.Render()

    def SetScene(self, scene):
        if self.renderer is None:
            raise ValueError("internal renderer instance has been moved.")
        scene = self.__SetScene(scene)
        self.renderer.SetScene(scene)

    def GetScene(self):
        if self.renderer is None:
            raise ValueError("internal renderer instance has been moved.")
        return Scene(None, None, None, None, self.renderer.GetScene())

    def __SetScene(self, scene):
        if scene is None:
            return PyBindMWIR.Scene()
        if type(scene) is not Scene:
            raise TypeError("scene must be of type Scene")
        tmp = scene.scene
        scene.scene = None
        return tmp