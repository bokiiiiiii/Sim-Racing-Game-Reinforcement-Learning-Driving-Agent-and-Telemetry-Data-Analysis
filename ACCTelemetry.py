import mmap
import struct
from typing import Any
from LocalParameters import LocalParametersClass


# @brief ACCTelemetry: Process ACC telemetry data from shared memory
class ACCTelemetry(LocalParametersClass):

    # @brief __init__: Initialize parameters and variables
    def __init__(self) -> None:
        print("ACC Telemetry initialized.")

        super().__init__()

        # Physics
        self.physics_shm_size: int = struct.calcsize(self.physics_layout)
        self.mmapPhysics: Any = None

        # Graphic
        self.graphic_shm_size: int = struct.calcsize(self.graphic_layout)
        self.mmapGraphic: Any = None

        # Static
        self.static_shm_size: int = struct.calcsize(self.static_layout)
        self.mmapStatic: Any = None

    # @brief start: Declare memory maps for physics, graphic, and static fields
    def start(self) -> None:
        print("ACC Telemetry started.")

        if not self.mmapPhysics:
            self.mmapPhysics = mmap.mmap(
                -1,
                self.physics_shm_size,
                "Local\\acpmf_physics",
                access=mmap.ACCESS_READ,
            )
        if not self.mmapGraphic:
            self.mmapGraphic = mmap.mmap(
                -1,
                self.graphic_shm_size,
                "Local\\acpmf_graphics",
                access=mmap.ACCESS_READ,
            )
        if not self.mmapStatic:
            self.mmapStatic = mmap.mmap(
                -1, self.static_shm_size, "Local\\acpmf_static", access=mmap.ACCESS_READ
            )

    # @brief getphysicsData: Process physics data from the memory map
    # @return: Physics data dictionary
    def getphysicsData(self) -> dict[str, Any]:
        self.mmapPhysics.seek(0)
        physics_rawData = self.mmapPhysics.read(self.physics_shm_size)
        physics_data = {}
        for index, value in enumerate(
            struct.unpack(self.physics_layout, physics_rawData)
        ):
            physics_data[self.physics_fields[index]] = value

        self._convertphysicsData(physics_data)
        return physics_data

    # @brief getgraphicData: Process graphic data from the memory map
    # @return: Graphic data dictionary
    def getgraphicData(self) -> dict[str, Any]:
        self.mmapGraphic.seek(0)
        graphic_rawData = self.mmapGraphic.read(self.graphic_shm_size)
        graphic_data = {}
        for index, value in enumerate(
            struct.unpack(self.graphic_layout, graphic_rawData)
        ):
            graphic_data[self.graphic_fields[index]] = value

        self._convertgraphicData(graphic_data)
        return graphic_data

    # @brief getstaticData: Process static data from the memory map
    # @return: Static data dictionary
    def getstaticData(self) -> dict[str, Any]:
        self.mmapStatic.seek(0)
        static_rawData = self.mmapStatic.read(self.static_shm_size)
        static_data = {}
        for index, value in enumerate(
            struct.unpack(self.static_layout, static_rawData)
        ):
            static_data[self.static_fields[index]] = value

        self._convertstaticData(static_data)
        return static_data

    # @brief stop: Stop ACC data telemetry monitoring
    def stop(self) -> None:
        print("ACC Telemetry stopped.")

        if self.mmapPhysics:
            self.mmapPhysics.close()
        self.mmapPhysics = None

        if self.mmapGraphic:
            self.mmapGraphic.close()
        self.mmapGraphic = None

        if self.mmapStatic:
            self.mmapStatic.close()
        self.mmapStatic = None

    # @brief _convertphysicsData: Convert physics data within the same category into aggregated data
    def _convertphysicsData(self, data: dict[str, Any]) -> None:
        # Make these conversions immediately while reading from shm
        # Combine X, Y, Z
        for newNameXYZ in self.physics_NameXYZList:
            data[newNameXYZ] = []
            for oldNameXYZ in [newNameXYZ + "X", newNameXYZ + "Y", newNameXYZ + "Z"]:
                data[newNameXYZ].append(data[oldNameXYZ])
                del data[oldNameXYZ]

        # Combine FL, FR, RL, RR
        for newNameFLFRRLRR in self.physics_NameFLFRRLRRList:
            data[newNameFLFRRLRR] = []
            for oldNameFLFRRLRR in [
                newNameFLFRRLRR + "FL",
                newNameFLFRRLRR + "FR",
                newNameFLFRRLRR + "RL",
                newNameFLFRRLRR + "RR",
            ]:
                data[newNameFLFRRLRR].append(data[oldNameFLFRRLRR])
                del data[oldNameFLFRRLRR]

    # @brief _convertgraphicData: Convert graphic data within the same category into aggregated data
    def _convertgraphicData(self, data: dict[str, Any]) -> None:
        # Make these conversions immediately while reading from shm
        # Combine x1, x2, ..., xn
        for newName in self.graphic_newNameList:
            data[newName] = []
            for oldName in getattr(self, f"{newName}_list"):
                data[newName].append(data[oldName])
                del data[oldName]

            # Decode bytes to char
            if newName in self.graphic_CharList:
                data_bytes = b"".join(data[newName])
                data_string = data_bytes.decode("utf-8", "ignore")
                data[newName] = data_string

    # @brief _convertstaticData: Convert graphic data within the same category into aggregated data
    def _convertstaticData(self, data: dict[str, Any]) -> None:
        # Make these conversions immediately while reading from shm
        # Combine x1, x2, ..., xn
        for newName in self.static_newNameList:
            data[newName] = []
            for oldName in getattr(self, f"{newName}_list"):
                data[newName].append(data[oldName])
                del data[oldName]

            # Decode bytes to char
            data_bytes = b"".join(data[newName])
            data_string = data_bytes.decode("utf-8", "ignore")
            data[newName] = data_string

        # Combine FL, FR, RL, RR
        for newNameFLFRRLRR in self.static_NameFLFRRLRRList:
            data[newNameFLFRRLRR] = []
            for oldNameFLFRRLRR in [
                newNameFLFRRLRR + "FL",
                newNameFLFRRLRR + "FR",
                newNameFLFRRLRR + "RL",
                newNameFLFRRLRR + "RR",
            ]:
                data[newNameFLFRRLRR].append(data[oldNameFLFRRLRR])
                del data[oldNameFLFRRLRR]

    # @brief getACCData: Combine physics, graphic, and static data dictionaries into a dictionary
    # @return: ACC Telemetry data dictionary
    def getACCData(self) -> dict[str, Any]:
        Current_physicsData = self.getphysicsData()
        Current_graphicData = self.getgraphicData()
        Current_staticData = self.getstaticData()
        Current_Data = {
            **Current_physicsData,
            **Current_graphicData,
            **Current_staticData,
        }
        return Current_Data
