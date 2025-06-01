# @brief LocalParametersClass: Define local parameters and variables
# @reference ACCSharedMemoryDocumentation
class LocalParametersClass:

    # @brief __init__: Initialize parameters and variables, including physics, graphic, and static fields
    def __init__(self) -> None:

        # * Physics
        physics_fields: str = (
            "packetId throttle brake fuel gear rpm steerAngle speedKmh "
            "velocityX velocityY velocityZ accGX accGY accGZ "
            "wheelSlipFL wheelSlipFR wheelSlipRL wheelSlipRR "
            "wheelLoadFL wheelLoadFR wheelLoadRL wheelLoadRR "
            "wheelsPressureFL wheelsPressureFR wheelsPressureRL wheelsPressureRR "
            "wheelAngularSpeedFL wheelAngularSpeedFR wheelAngularSpeedRL wheelAngularSpeedRR "
            "TyrewearFL TyrewearFR TyrewearRL TyrewearRR "
            "tyreDirtyLevelFL tyreDirtyLevelFR tyreDirtyLevelRL tyreDirtyLevelRR "
            "TyreCoreTempFL TyreCoreTempFR TyreCoreTempRL TyreCoreTempRR "
            "camberRADFL camberRADFR camberRADRL camberRADRR "
            "suspensionTravelFL suspensionTravelFR suspensionTravelRL suspensionTravelRR "
            "drs tcAction heading pitch roll cgHeight "
            "carDamagefront carDamagerear carDamageleft carDamageright carDamagecentre "
            "numberOfTyresOut pitLimiterOn absAction kersCharge kersInput automat "
            "rideHeightfront rideHeightrear turboBoost ballast airDensity airTemp roadTemp "
            "localAngularVelX localAngularVelY localAngularVelZ "
            "finalFF performanceMeter engineBrake ersRecoveryLevel ersPowerLevel ersHeatCharging ersIsCharging "
            "kersCurrentKJ drsAvailable drsEnabled "
            "brakeTempFL brakeTempFR brakeTempRL brakeTempRR "
            "clutch tyreTempI1 tyreTempI2 tyreTempI3 tyreTempI4 "
            "tyreTempM1 tyreTempM2 tyreTempM3 tyreTempM4 "
            "tyreTempO1 tyreTempO2 tyreTempO3 tyreTempO4 "
            "isAIControlled tyreContactPointFLX tyreContactPointFLY tyreContactPointFLZ "
            "tyreContactPointFRX tyreContactPointFRY tyreContactPointFRZ "
            "tyreContactPointRLX tyreContactPointRLY tyreContactPointRLZ "
            "tyreContactPointRRX tyreContactPointRRY tyreContactPointRRZ "
            "tyreContactNormalFLX tyreContactNormalFLY tyreContactNormalFLZ "
            "tyreContactNormalFRX tyreContactNormalFRY tyreContactNormalFRZ "
            "tyreContactNormalRLX tyreContactNormalRLY tyreContactNormalRLZ "
            "tyreContactNormalRRX tyreContactNormalRRY tyreContactNormalRRZ "
            "tyreContactHeadingFLX tyreContactHeadingFLY tyreContactHeadingFLZ "
            "tyreContactHeadingFRX tyreContactHeadingFRY tyreContactHeadingFRZ "
            "tyreContactHeadingRLX tyreContactHeadingRLY tyreContactHeadingRLZ "
            "tyreContactHeadingRRX tyreContactHeadingRRY tyreContactHeadingRRZ "
            "brakeBias localVelocityX localVelocityY localVelocityZ P2PActivation P2PStatus currentMaxRpm "
            "mzFL mzFR mzRL mzRR fxFL fxFR fxRL fxRR fyFL fyFR fyRL fyRR "
            "slipRatioFL slipRatioFR slipRatioRL slipRatioRR "
            "slipAngleFL slipAngleFR slipAngleRL slipAngleRR "
            "tcinAction absInAction suspensionDamageFL suspensionDamageFR suspensionDamageRL suspensionDamageRR "
            "tyreTempFL tyreTempFR tyreTempRL tyreTempRR waterTemp "
            "brakePressureFL brakePressureFR brakePressureRL brakePressureRR "
            "frontBrakeCompound rearBrakeCompound padLifeFL padLifeFR padLifeRL padLifeRR "
            "discLifeFL discLifeFR discLifeRL discLifeRR "
        )
        self.physics_fields: list[str] = physics_fields.replace("  ", " ").split(" ")

        physics_layout: str = (
            "ifffiiff"
            "3f3f4f"
            "4f4f4f4f4f4f4f4f"
            "ffffff5f"
            "iifff"
            "i2ffffff3fff"
            "iiiiifii"
            "4ff4f4f4fi"
            "12f12f12f"
            "f3fiif"
            "4f4f4f4f4f"
            "ii4f4ff4f"
            "ii4f4f"
        )
        self.physics_layout: str = physics_layout

        self.physics_NameXYZList: list[str] = [
            "velocity",
            "accG",
            "localAngularVel",
            "tyreContactPointFL",
            "tyreContactPointFR",
            "tyreContactPointRL",
            "tyreContactPointRR",
            "tyreContactNormalFL",
            "tyreContactNormalFR",
            "tyreContactNormalRL",
            "tyreContactNormalRR",
            "tyreContactHeadingFL",
            "tyreContactHeadingFR",
            "tyreContactHeadingRL",
            "tyreContactHeadingRR",
            "localVelocity",
        ]

        self.physics_NameFLFRRLRRList: list[str] = [
            "wheelSlip",
            "wheelLoad",
            "wheelsPressure",
            "wheelAngularSpeed",
            "Tyrewear",
            "tyreDirtyLevel",
            "TyreCoreTemp",
            "camberRAD",
            "suspensionTravel",
            "brakeTemp",
            "tyreContactPoint",
            "tyreContactNormal",
            "tyreContactHeading",
            "mz",
            "fx",
            "fy",
            "slipRatio",
            "slipAngle",
            "suspensionDamage",
            "tyreTemp",
            "brakePressure",
            "padLife",
            "discLife",
        ]

        # * Graphic
        graphic_partial_fields: str = (
            "packetId status session "
            # currentTime lastTime bestTime split
            "completedLaps position iCurrentTime iLastTime iBestTime "
            "sessionTimeLeft distanceTraveled isInPit currentSectorIndex lastSectorTime numberOfLaps "
            # tyreCompound
            "replayTimeMultiplier normalizedCarPosition activeCars "
            # carCoordinates carID
            "playerCarID penaltyTime flag penalty idealLineOn isInPitLane surfaceGrip "
            "mandatoryPitDone windSpeed windDirection isSetupMenuVisible "
            "mainDisplayIndex secondaryDisplyIndex TC TCCUT EngineMap ABS fuelXLap "
            "rainLights flashingLights lightsStage exhaustTemperature "
            "wiperLV driverStintTotalTimeLeft driverStintTimeLeft rainTyres sessionIndex usedFuel "
            # deltaLapTime
            "iDeltaLapTime "
            # estimatedLapTime
            "iEstimatedLapTime isDeltaPositive iSplit isValidLap fuelEstimatedLaps "
            # trackStatus
            "missingMandatoryPits Clock directionLightsLeft directionLightsRight "
            "GlobalYellow GlobalYellow1 GlobalYellow2 GlobalYellow3 "
            "GlobalWhite GlobalGreen GlobalChequered GlobalRed mfdTyreSet mfdFuelToAdd "
            "mfdTyrePressureLF mfdTyrePressureRF mfdTyrePressureLR mfdTyrePressureRR "
            "trackGripStatus rainIntensity rainIntensityIn10min rainIntensityIn30min "
            "currentTyreSet strategyTyreSet gapAhead gapBehind"
        )
        graphic_fields: list[str] = graphic_partial_fields.replace("  ", " ").split(" ")

        graphic_lists_name_length_index: dict[str, list[int]] = {
            # 'name': [length, index]
            "currentTime": [30, 3],
            "lastTime": [30, 33],
            "bestTime": [30, 63],
            "split": [30, 93],
            "tyreCompound": [66, 134],
            "carCoordinates": [180, 203],
            "carID": [60, 383],
            "deltaLapTime": [30, 471],
            "estimatedLapTime": [30, 502],
            "trackStatus": [66, 537],
        }

        # Create lists for array data
        graphic_lists: dict[str, tuple[list[str], int]] = {
            f"{name}_list": ([f"{name}{i}" for i in range(1, value[0] + 1)], value[1])
            for name, value in graphic_lists_name_length_index.items()
        }

        for name, value in graphic_lists.items():
            setattr(self, name, value[0])  # Define self.lists
            graphic_fields[value[1] : value[1]] = value[0]  # Insert lists into fields

        self.graphic_fields: list[str] = graphic_fields

        graphic_layout: str = (
            "iii30c30c30c30c"  # sizeof(c_wchar) = sizeof(c_char) * 2
            "iiiiiffiiii"
            "66cffi180f60i"
            "ifiiiififfi"
            "iiiiiif"
            "iiifiiiiif"
            "30ci30ciiiif66c"
            "ifiiiiiiiiiiiff"
            "fffiiiiiiii"
        )
        self.graphic_layout: str = graphic_layout

        self.graphic_newNameList: list[str] = [
            lists[0] for lists in graphic_lists_name_length_index.items()
        ]

        self.graphic_CharList: list[str] = [
            "currentTime",
            "lastTime",
            "bestTime",
            "split",
            "tyreCompound",
            "deltaLapTime",
            "estimatedLapTime",
            "trackStatus",
        ]

        # * Static
        static_partial_fields: str = (
            # smVersion acVersion
            "numberOfSessions numCars "
            # carModel track playerName playerSurname playerNick
            "sectorCount maxTorque maxPower maxRpm maxFuel "
            "suspensionMaxTravelFL suspensionMaxTravelFR suspensionMaxTravelRL suspensionMaxTravelRR "
            "tyreRadiusFL tyreRadiusFR tyreRadiusRL tyreRadiusRR "
            "maxTurboBoost deprecated_1 deprecated_2 penaltiesEnabled "
            "aidFuelRate aidTireRate aidMechanicalDamage AllowTyreBlankets aidStability aidAutoclutch aidAutoBlip "
            "hasDRS hasERS hasKERS KersMaxJ engineBrakeSettingsCount ersPowerControllerCount trackSplineLength "
            # trackConfiguration
            "ersMaxJ isTimedRace hasExtraLap "
            # carSkin
            "reversedGridPositions PitWindowStart PitWindowEnd isOnline"
            # dryTyresName wetTyresName
        )
        static_fields: list[str] = static_partial_fields.replace("  ", " ").split(" ")

        static_lists_name_length_index: dict[str, list[int]] = {
            # 'name': [length, index]
            "smVersion": [30, 0],
            "acVersion": [30, 30],
            "carModel": [66, 62],
            "track": [66, 128],
            "playerName": [66, 194],
            "playerSurname": [66, 260],
            "playerNick": [66, 326],
            "trackConfiguration": [66, 423],
            "carSkin": [66, 492],
            "dryTyresName": [66, 562],
            "wetTyresName": [66, 628],
        }

        # Create lists for array data
        static_lists: dict[str, tuple[list[str], int]] = {
            f"{name}_list": ([f"{name}{i}" for i in range(1, value[0] + 1)], value[1])
            for name, value in static_lists_name_length_index.items()
        }

        for name, value in static_lists.items():
            setattr(self, name, value[0])  # Define self.lists
            static_fields[value[1] : value[1]] = value[0]  # Insert lists into fields

        self.static_fields: list[str] = static_fields

        static_layout: str = (
            "30c30cii"
            "66c66c66c66c66c"
            "iffif4f4f"
            "fffifffffiiiiifiif"
            "66cfii66c"
            "iiii66c66c"
        )
        self.static_layout: str = static_layout

        self.static_newNameList: list[str] = [
            lists[0] for lists in static_lists_name_length_index.items()
        ]

        self.static_CharList: list[str] = (
            self.static_newNameList
        )  # All newNameList is CharList

        self.static_NameFLFRRLRRList: list[str] = ["suspensionMaxTravel", "tyreRadius"]
