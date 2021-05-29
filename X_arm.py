#!/home/lab/.conda/envs/lab502/bin/python
# -*- coding: UTF-8

import serial
import time
import struct
import keyboard

class X_arm:
    def __init__(self):
        self.ser = None
        self.head = b'\xf5\x5f'
        self.motorId = b'\x01'
        self.instruction = b'\x01'
        self.speedPercent = 0.5
        self.andCrc = 255
        self.xorCrc = 255
        self.angle = 0.0
        self.calibration_manual_on=0

    def __del__(self):
        self.ser.close()

    def ConvertFixedIntegerToComplement(self, fixedInterger):
        return bin(fixedInterger)[2:]

    def ConvertFixedDecimalToComplement(self, fixedDecimal):
        fixedpoint = int(fixedDecimal) / (10.0 ** len(fixedDecimal))
        s = ''
        while fixedDecimal != 1.0 and len(s) < 23:
            fixedpoint = fixedpoint * 2.0
            s += str(fixedpoint)[0]
            fixedpoint = fixedpoint if str(fixedpoint)[0] == '0' else fixedpoint - 1.0
        return s

    def ConvertToExponentMarker(self, number):

        return bin(number + 127)[2:].zfill(8)

    def ConvertToFloat(self, floatingPoint):

        floatingPoint = float(floatingPoint)
        floatingPointString = str(floatingPoint)
        if floatingPointString.find('-') != -1:
            sign = '1'
            floatingPointString = floatingPointString[1:]
        else:
            sign = '0'
        l = floatingPointString.split('.')
        front = self.ConvertFixedIntegerToComplement(int(l[0]))
        rear = self.ConvertFixedDecimalToComplement(l[1])
        floatingPointString = front + '.' + rear
        relativePos = floatingPointString.find('.') - floatingPointString.find('1')
        if relativePos > 0:
            exponet = self.ConvertToExponentMarker(relativePos - 1)
            mantissa = floatingPointString[
                       floatingPointString.find('1') + 1: floatingPointString.find('.')] + floatingPointString[
                                                                                           floatingPointString.find(
                                                                                               '.') + 1:]
        else:
            exponet = self.ConvertToExponentMarker(relativePos)
            mantissa = floatingPointString[floatingPointString.find('1') + 1:]
        mantissa = mantissa[:23] + '0' * (23 - len(mantissa))
        floatingPointString = '0b' + sign + exponet + mantissa
        return int(int(floatingPointString, 2)).to_bytes(length=4, byteorder='little', signed=False)

    # print(ConvertToFloat(-5.56))

    def print_hex(self, bytes):
        l = [hex(int(i)) for i in bytes]
        print(" ".join(l))

    def setJointAngle(self, id, targetAngle, speedPercent):

        self.angle = targetAngle
        self.speedPercent = speedPercent
        self.instruction = b'\x01'
        data = self.head + int(id).to_bytes(length=1, byteorder='little', signed=False) \
               + self.instruction \
               + self.ConvertToFloat(self.angle) \
               + self.ConvertToFloat(self.speedPercent)
        for iterating_var in data:
            self.andCrc = int(self.andCrc) and iterating_var
            self.xorCrc = int(self.xorCrc) ^ iterating_var
        data = data + self.andCrc.to_bytes(length=1, byteorder='little', signed=False) + self.xorCrc.to_bytes(length=1,
                                                                                                              byteorder='little',
                                                                                                              signed=False)
        # self.print_hex(data)
        self.andCrc = 255
        self.xorCrc = 255
        if self.ser.is_open and self.ser is not None:
            self.ser.write(data)
        time.sleep(0.5)
    def powerOff(self,isOn):
        self.instruction = b'\x02'
        data = self.head + b'\x08' \
               + self.instruction \
               + int(isOn).to_bytes(length=1, byteorder='little', signed=False) \
               + b'\xff' + b'\xff' + b'\xff' + b'\xff' + b'\xff' + b'\xff' + b'\xff'
        for iterating_var in data:
            self.andCrc = int(self.andCrc) and iterating_var
            self.xorCrc = int(self.xorCrc) ^ iterating_var
        data = data + self.andCrc.to_bytes(length=1, byteorder='little', signed=False) + self.xorCrc.to_bytes(length=1,
                                                                                                              byteorder='little',
                                                                                                              signed=False)
        self.print_hex(data)
        self.andCrc = 255
        self.xorCrc = 255
        if self.ser.is_open and self.ser is not None:
            self.ser.write(data)
        time.sleep(0.5)
        if isOn==1:
            print("entering lose power mode ")
        elif isOn==0:
            print("quitting lose power mode")



    def Open_port(self, comNumber):
        self.ser = serial.Serial(comNumber, 115200)
        if self.ser.is_open:
            print("open success")
    def getStates(self):
        is_exit = False
        motor_states={
            'J1': 0.0,
            'J2': 0.0,
            'J3': 0.0,
            'J4': 0.0,
            'J5': 0.0,
            'J6': 0.0,
        }
        self.instruction = b'\x03'
        data = self.head + b'\x09' \
               + self.instruction \
               + b'\x01' \
               + b'\xff' + b'\xff' + b'\xff' + b'\xff' + b'\xff' + b'\xff' + b'\xff'
        for iterating_var in data:
            self.andCrc = int(self.andCrc) and iterating_var
            self.xorCrc = int(self.xorCrc) ^ iterating_var
        data = data + self.andCrc.to_bytes(length=1, byteorder='little', signed=False) + self.xorCrc.to_bytes(length=1,
                                                                                                              byteorder='little',
                                                                                                              signed=False)
        self.print_hex(data)
        self.andCrc = 255
        self.xorCrc = 255

        if self.ser.is_open and self.ser is not None:
            self.ser.write(data)
        time.sleep(1)
        while not is_exit:
            count = self.ser.inWaiting()
            if count > 20:
                rec_str = self.ser.read(count)
                #data_bytes = data_bytes + rec_str
                presentPosition=struct.unpack('<ffffff',rec_str)
                # for i in range(0,6):
                #     print(int(presentPosition[i]))
                motor_states['J1'] = int(presentPosition[0])
                motor_states['J2']=int(presentPosition[1])
                motor_states['J3'] = int(presentPosition[2])
                motor_states['J4'] = int(presentPosition[3])
                motor_states['J5']=int(presentPosition[4])
                motor_states['J6'] = int(presentPosition[5])
                for motorID in motor_states:
                    print(motorID+":"+str(motor_states[motorID]))
                is_exit=True

        time.sleep(1)

    def calibration_manual(self):
        self.powerOff(1)
        print("entering calibration mode")
        print("1 manual mode")
        print("2 press m ")
        keyboard.add_hotkey('m', self.getStates)
        keyboard.wait('ctrl+enter')
        self.powerOff(0)
        print("over")