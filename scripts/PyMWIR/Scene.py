import torch
from . import PyBindMWIR
from . import Mesh
from . import Antenna
from . import Signal

class Scene:
    def __init__(self, mesh = None, senders = None, receivers = None, signal = None):
        mesh = self.__SetMesh(mesh)
        senders = self.__SetSenders(senders)
        receivers = self.__SetReceivers(receivers)
        signal = self.__SetSignal(signal)
        self.scene = PyBindMWIR.Scene(mesh, senders, receivers, signal)
  
    def SetMesh(self, mesh):
        if self.scene is None:
            raise ValueError("Scene ownership has been transferred")
        mesh = self.__SetMesh(mesh)
        self.scene.SetMesh(mesh)

    def SetSenders(self, senders):
        if self.scene is None:
            raise ValueError("Scene ownership has been transferred")
        senders = self.__SetSenders(senders)
        self.scene.SetSenders(senders)

    def SetReceivers(self, receivers):
        if self.scene is None:
            raise ValueError("Scene ownership has been transferred")
        receivers = self.__SetReceivers(receivers)
        self.scene.SetReceivers(receivers)

    def SetSignal(self, signal):
        if self.scene is None:
            raise ValueError("Scene ownership has been transferred")
        signal = self.__SetSignal(signal)
        self.scene.SetSignal(signal)

    def GetMesh(self):
        if self.scene is None:
            raise ValueError("Scene ownership has been transferred")
        tmp = Mesh(None, self.scene.GetMesh())
        return tmp

    def GetSenders(self):
        if self.scene is None:
            raise ValueError("Scene ownership has been transferred")
        tmp = self.scene.GetSenders()
        senders = []
        for i in range(len(tmp)):
            senders.append(Antenna(None, None, None, None, tmp[i]))
        return senders

    def GetReceivers(self):
        if self.scene is None:
            raise ValueError("Scene ownership has been transferred")
        tmp = self.scene.GetReceivers()
        receivers = []
        for i in range(len(tmp)):
            receivers.append(Antenna(None, None, None, None, tmp[i]))
        return receivers

    def GetSignal(self):
        if self.scene is None:
            raise ValueError("Scene ownership has been transferred")
        tmp = Signal(None, None, self.scene.GetSignal())
        return tmp

    def __SetMesh(self, mesh):
        if mesh is None:
            return PyBindMWIR.Mesh()
        if type(mesh) is not Mesh:
            raise TypeError("mesh must be of type Mesh")
        return mesh.mesh

    def __SetSenders(self, senders):
        if senders is None:
            return [PyBindMWIR.Antenna()]
           
        if not all(type(sender) is Antenna for sender in senders):
            raise TypeError("senders must be a list of Antennas")
        __senders = []
        for sender in senders:
            __senders.append(sender.antenna)
        return __senders

    def __SetReceivers(self, receivers):
        if receivers is None:
            return [PyBindMWIR.Antenna()]
        if not all(type(receiver) is Antenna for receiver in receivers):
            raise TypeError("receivers must be a list of Antennas")
        __receivers = []
        for receiver in receivers:
            __receivers.append(receiver.antenna)
        return __receivers

    def __SetSignal(self, signal):
        if signal is None:
            return PyBindMWIR.Signal()
        if type(signal) is not Signal:
            raise TypeError("signal must be of type Signal")
        return signal.signal

     