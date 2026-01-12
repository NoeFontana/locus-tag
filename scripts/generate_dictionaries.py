#!/usr/bin/env python3
"""Generate Rust dictionary files from AprilTag C sources.

This script parses the official UMich AprilTag sources and generates
Rust source files with all tag codes. It handles the bit ordering
conversion from UMich's spiral ordering to our row-major ordering.

Usage:
    python scripts/generate_dictionaries.py
"""

import sys
from pathlib import Path

# Add scripts directory to path to import generated data modules
sys.path.append(str(Path(__file__).parent))

try:
    import apriltag_41h12_data
except ImportError:
    print("Warning: apriltag_41h12_data.py not found. 41h12 will be skipped.")
    apriltag_41h12_data = None


# UMich AprilTag source URLs (for reference)
# https://github.com/AprilRobotics/apriltag/blob/master/tag36h11.c
# https://github.com/AprilRobotics/apriltag/blob/master/tag16h5.c

UMICH_LICENSE = """\
// Copyright (C) 2013-2016, The Regents of The University of Michigan.
// All rights reserved.
// This software was developed in the APRIL Robotics Lab under the
// direction of Edwin Olson, ebolson@umich.edu. This software may be
// available under alternative licensing terms; contact the address above.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# All 587 codes for AprilTag 36h11 (from UMich source)
# These use UMich's spiral bit ordering
APRILTAG_36H11_CODES_SPIRAL = [
    0x0000000D7E00984B,
    0x0000000DDA664CA7,
    0x0000000DC4A1C821,
    0x0000000E17B470E9,
    0x0000000EF91D01B1,
    0x0000000F429CDD73,
    0x000000005DA29225,
    0x00000001106CBA43,
    0x0000000223BED79D,
    0x000000021F51213C,
    0x000000033EB19CA6,
    0x00000003F76EB0F8,
    0x0000000469A97414,
    0x000000045DCFE0B0,
    0x00000004A6465F72,
    0x00000005180DDB96,
    0x00000005EB946B4E,
    0x000000068A7CC2EC,
    0x00000006F0BA2652,
    0x000000078765559D,
    0x000000087B83D129,
    0x000000086CC4A5C5,
    0x00000008B64DF90F,
    0x00000009C577B611,
    0x0000000A3810F2F5,
    0x0000000AF4D75B83,
    0x0000000B59A03FEF,
    0x0000000BB1096F85,
    0x0000000D1B92FC76,
    0x0000000D0DD509D2,
    0x0000000E2CFDA160,
    0x00000002FF497C63,
    0x000000047240671B,
    0x00000005047A2E55,
    0x0000000635CA87C7,
    0x0000000691254166,
    0x000000068F43D94A,
    0x00000006EF24BDB6,
    0x00000008CDD8F886,
    0x00000009DE96B718,
    0x0000000AFF6E5A8A,
    0x0000000BAE46F029,
    0x0000000D225B6D59,
    0x0000000DF8BA8C01,
    0x0000000E3744A22F,
    0x0000000FBB59375D,
    0x000000018A916828,
    0x000000022F29C1BA,
    0x0000000286887D58,
    0x000000041392322E,
    0x000000075D18ECD1,
    0x000000087C302743,
    0x00000008C6317BA9,
    0x00000009E40F36D7,
    0x0000000C0E5A806A,
    0x0000000CC78CB87C,
    0x000000012D2F2D01,
    0x0000000379F36A21,
    0x00000006973F59AC,
    0x00000007789EA9F4,
    0x00000008F1C73E84,
    0x00000008DD287A20,
    0x000000094A4EEE4C,
    0x0000000A455379B5,
    0x0000000A9E92987D,
    0x0000000BD25CB40B,
    0x0000000BE98D3582,
    0x0000000D3D5972B2,
    0x000000014C53D7C7,
    0x00000004F1796936,
    0x00000004E71FED1A,
    0x000000066D46FAE0,
    0x0000000A55ABB933,
    0x0000000EBEE1ACCCA,
    0x00000001AD4BA6A4,
    0x0000000305B17571,
    0x0000000553611351,
    0x000000059CA62775,
    0x00000007819CB6A1,
    0x0000000EDB7BC9EB,
    0x00000005B2694212,
    0x000000072E12D185,
    0x0000000ED6152E2C,
    0x00000005BCDADBF3,
    0x000000078E0AA0C6,
    0x0000000C60A0B909,
    0x0000000EF9A34B0D,
    0x00000003988A6621A,
    0x0000000A8A27C944,
    0x00000004B564304E,
    0x000000052902B4E2,
    0x0000000857280B56,
    0x0000000A91B2C84B,
    0x0000000E91DF939B,
    0x00000001FA405F28,
    0x000000023793AB86,
    0x000000068C17729F,
    0x00000009FBF3B840,
    0x000000036922413C,
    0x00000004EB5F946E,
    0x0000000533FE2404,
    0x000000063DE7D35E,
    0x0000000925EDDC72,
    0x000000099B8B3896,
    0x0000000AACE4C708,
    0x0000000C22994AF0,
    0x00000008F1EAE41B,
    0x0000000D95FB486C,
    0x000000013FB77857,
    0x00000004FE0983A3,
    0x0000000D559BF8A9,
    0x0000000E1855D78D,
    0x0000000FEC8DAAAD,
    0x000000071ECB6D95,
    0x0000000DC9E50E4C,
    0x0000000CA3A4C259,
    0x00000007400D12BBF,
    0x0000000AEEDD18E0,
    0x0000000B509B9C8E,
    0x00000005232FEA1C,
    0x000000019282D18B,
    0x000000076C22D67B,
    0x0000000936BEB34B,
    0x000000008A5EA8DD,
    0x0000000679EADC28,
    0x0000000A08E119C5,
    0x000000020A6E3E24,
    0x00000007EAB9C239,
    0x000000096632C32E,
    0x00000004700D06E44,
    0x00000008A70212FB,
    0x00000000A7E4251B,
    0x00000009EC762CC0,
    0x0000000D8A3A1F48,
    0x0000000DB680F346,
    0x00000004A1E93A9D,
    0x0000000638DDC04F,
    0x00000004C2FCC993,
    0x000000001EF28C95,
    0x0000000BF0D9792D,
    0x00000006D27557C3,
    0x0000000623F977F4,
    0x000000035B43BE57,
    0x0000000BB0C428D5,
    0x0000000A6F01474D,
    0x00000005A70C9749,
    0x000000020DDABC3B,
    0x00000002EABD78CF,
    0x000000090AA18F88,
    0x0000000A9EA89350,
    0x00000003CDB39B22,
    0x0000000839A08F34,
    0x0000000169BB814E,
    0x00000001A575AB08,
    0x0000000A04D3D5A2,
    0x0000000BF7902F2B,
    0x0000000095A5E65C,
    0x000000092E8FCE94,
    0x000000067EF48D12,
    0x00000006400DBCAC,
    0x0000000B12D8FB9F,
    0x00000000347F45D3,
    0x0000000B35826F56,
    0x0000000C546AC6E4,
    0x000000081CC35B66,
    0x000000041D14BD57,
    0x00000000C052B168,
    0x00000007D6CE5018,
    0x0000000AB4ED5EDE,
    0x00000005AF817119,
    0x0000000D1454B182,
    0x00000002BADB090B,
    0x000000003FCB4C0C,
    0x00000002F1C28FD8,
    0x000000093608C6F7,
    0x00000004C93BA2B5,
    0x000000007D950A5D,
    0x0000000E54B3D3FC,
    0x000000015560CF9D,
    0x0000000189E4958A,
    0x000000062140E9D2,
    0x0000000723BC1CDB,
    0x00000002063F26FA,
    0x0000000FA08AB19F,
    0x00000007955641DB,
    0x0000000646B01DAA,
    0x000000071CD427CC,
    0x000000009A42F7D4,
    0x0000000717EDC643,
    0x000000015EB94367,
    0x00000008392E6BB2,
    0x0000000832408542,
    0x00000002B9B874BE,
    0x0000000B21F4730D,
    0x0000000B5D8F24C9,
    0x00000007DBAF6931,
    0x00000001B4E33629,
    0x000000013452E710,
    0x0000000E974AF612,
    0x00000001DF61D29A,
    0x000000099F2532AD,
    0x0000000E50EC71B4,
    0x00000005DF0A36E8,
    0x00000004934E4CEA,
    0x0000000E34A0B4BD,
    0x0000000B7B26B588,
    0x00000000F255118D,
    0x0000000D0C8FA31E,
    0x000000006A50C94F,
    0x0000000F28AA9F06,
    0x0000000131D194D8,
    0x0000000622E3DA79,
    0x0000000AC7478303,
    0x0000000C8F2521D7,
    0x00000006C9C881F5,
    0x000000049E38B60A,
    0x0000000513D8DF65,
    0x0000000D7C2B0785,
    0x00000009F6F9D75A,
    0x00000009F6966020,
    0x00000001E1A54E33,
    0x0000000C04D63419,
    0x0000000946E04CD7,
    0x00000001BDAC5902,
    0x000000056469B830,
    0x0000000FFAD59569,
    0x000000086970E7D8,
    0x00000008A4B41E12,
    0x0000000AD4688E3B,
    0x000000085F8F5DF4,
    0x0000000D833A0893,
    0x00000002A36FDD7C,
    0x0000000D6A857CF2,
    0x00000008829BC35C,
    0x00000005E50D79BC,
    0x0000000FBB8035E4,
    0x0000000C1A95BEBF,
    0x0000000036B0BAF8,
    0x0000000E0DA964EA,
    0x0000000B6483689B,
    0x00000007C8E2F4C1,
    0x00000005B856A23B,
    0x00000002FC183995,
    0x0000000E914B6D70,
    0x0000000B31041969,
    0x00000001BB478493,
    0x0000000063E2B456,
    0x0000000F2A082B9C,
    0x00000008E5E646EA,
    0x000000008172F8F6,
    0x00000000DACD923E,
    0x0000000E5DCF0E2E,
    0x0000000BF9446BAE,
    0x00000004822D50D1,
    0x000000026E710BF5,
    0x0000000B90BA2A24,
    0x0000000F3B25AA73,
    0x0000000809AD589B,
    0x000000094CC1E254,
    0x00000005334A3ADB,
    0x0000000592886B2F,
    0x0000000BF64704AA,
    0x0000000566DBF24C,
    0x000000072203E692,
    0x000000064E61E809,
    0x0000000D7259AAD6,
    0x00000007B924AEDC,
    0x00000002DF2184E8,
    0x0000000353D1ECA7,
    0x0000000FCE30D7CE,
    0x0000000F7B0F436E,
    0x000000057E8D8F68,
    0x00000008C79E60DB,
    0x00000009C8362B2B,
    0x000000063A5804F2,
    0x00000009298353DC,
    0x00000006F98A71C8,
    0x0000000A5731F693,
    0x000000021CA5C870,
    0x00000001C2107FD3,
    0x00000006181F6C39,
    0x000000019E574304,
    0x0000000329937606,
    0x0000000043D5C70D,
    0x00000009B18FF162,
    0x00000008E2CCFEBF,
    0x000000072B7B9B54,
    0x00000009B71F4F3C,
    0x0000000935D7393E,
    0x000000065938881A,
    0x00000006A5BD6F2D,
    0x0000000A19783306,
    0x0000000E6472F4D7,
    0x000000081163DF5A,
    0x0000000A838E1CBD,
    0x0000000982748477,
    0x0000000050C54FEB,
    0x00000000D82FBB58,
    0x00000002C4C72799,
    0x000000097D259AD6,
    0x000000022D9A43ED,
    0x0000000FDB162A9F,
    0x00000000CB4A727D,
    0x00000004FAE2E371,
    0x0000000535B5BE8B,
    0x000000048795908A,
    0x0000000CE7C18962,
    0x00000004EA154D80,
    0x000000050C064889,
    0x00000008D97FC75D,
    0x0000000C8BD9EC61,
    0x000000083EE8E8BB,
    0x0000000C8431419A,
    0x00000001AA78079D,
    0x00000008111AA4A5,
    0x0000000DFA3A69FE,
    0x000000051630D83F,
    0x00000002D930FB3F,
    0x00000002133116E5,
    0x0000000AE5395522,
    0x0000000BC07A4E8A,
    0x000000057BF08BA0,
    0x00000006CB18036A,
    0x0000000F0E2E4B75,
    0x00000003EB692B6F,
    0x0000000D8178A3FA,
    0x0000000238CCE6A6,
    0x0000000E97D5CDD7,
    0x0000000FE10D8D5E,
    0x0000000B39584A1D,
    0x0000000CA03536FD,
    0x0000000AA61F3998,
    0x000000072FF23EC2,
    0x000000015AA7D770,
    0x000000057A3A1282,
    0x0000000D1F3902DC,
    0x00000006554C9388,
    0x0000000FD01283C7,
    0x0000000E8BAA42C5,
    0x000000072CEE6ADF,
    0x0000000F6614B3FA,
    0x000000095C3778A2,
    0x00000007DA4CEA7A,
    0x0000000D18A5912C,
    0x0000000D116426E5,
    0x000000027C17BC1C,
    0x0000000B95B53BC1,
    0x0000000C8F937A05,
    0x0000000ED220C9BD,
    0x00000000C97D72AB,
    0x00000008FB1217AE,
    0x000000025CA8A5A1,
    0x0000000B261B871B,
    0x00000001BEF0A056,
    0x0000000806A51179,
    0x0000000EED249145,
    0x00000003F82AECEB,
    0x0000000CC56E9ACF,
    0x00000002E78D01EB,
    0x0000000102CEE17F,
    0x000000037CAAD3D5,
    0x000000016AC5B1EE,
    0x00000002AF164ECC,
    0x0000000D4CD81DC9,
    0x000000012263A7E7,
    0x000000057AC7D117,
    0x00000009391D9740,
    0x00000007AEDAA77F,
    0x00000009675A3C72,
    0x0000000277F25191,
    0x0000000EBB6E64B9,
    0x00000007AD3EF747,
    0x000000012759B181,
    0x0000000948257D4D,
    0x0000000B63A850F6,
    0x00000003A52A8F75,
    0x00000004A019532C,
    0x0000000A021A7529,
    0x0000000CC661876D,
    0x00000004085AFD05,
    0x0000000E7048E089,
    0x00000003F979CDC6,
    0x0000000D9DA9071B,
    0x0000000ED2FC5B68,
    0x000000079D64C3A1,
    0x0000000FD44E2361,
    0x00000008EEA46A74,
    0x000000042233B9C2,
    0x0000000AE4D1765D,
    0x00000007303A094C,
    0x00000002D7033ABE,
    0x00000003DCC2B0B4,
    0x00000000F0967D09,
    0x000000006F0CD7DE,
    0x000000009807ACA0,
    0x00000003A295CAD3,
    0x00000002B106B202,
    0x00000003F38A828E,
    0x000000078AF46596,
    0x0000000BDA2DC713,
    0x00000009A8C8C9D9,
    0x00000006A0F2DDCE,
    0x0000000A76AF6FE2,
    0x0000000086F66FA4,
    0x0000000D52D63F8D,
    0x000000089F7A6E73,
    0x0000000CC6B23362,
    0x0000000B4EBF3C39,
    0x0000000564F300FA,
    0x0000000E8DE3A706,
    0x000000079A033B61,
    0x0000000765E160C5,
    0x0000000A266A4F85,
    0x0000000A68C38C24,
    0x0000000DCA0711FB,
    0x000000085FBA85BA,
    0x000000037A207B46,
    0x0000000158FCC4D0,
    0x00000000569D79B3,
    0x00000007B1A25555,
    0x0000000A8AE22468,
    0x00000007C592BDFD,
    0x00000000C59A5F66,
    0x0000000B1115DAA3,
    0x0000000F17C87177,
    0x00000006769D766B,
    0x00000002B637356D,
    0x000000013D8685AC,
    0x0000000F24CB6EC0,
    0x00000000BD0B56D1,
    0x000000042FF0E26D,
    0x0000000B41609267,
    0x000000096F9518AF,
    0x0000000C56F96636,
    0x00000004A8E10349,
    0x0000000863512171,
    0x0000000EA455D86C,
    0x0000000BD0E25279,
    0x0000000E65E3F761,
    0x000000036C84A922,
    0x000000085FD1B38F,
    0x0000000657C91539,
    0x000000015033FE04,
    0x000000009051C921,
    0x0000000AB27D80D8,
    0x0000000F92F7D0A1,
    0x00000008EB6BB737,
    0x000000010B5B0F63,
    0x00000006C9C7AD63,
    0x0000000F66FE70AE,
    0x0000000CA579BD92,
    0x0000000956198E4D,
    0x000000029E4405E5,
    0x0000000E44EB885C,
    0x000000041612456C,
    0x0000000EA45E0ABF,
    0x0000000D326529BD,
    0x00000007B2C33CEF,
    0x000000080BC9B558,
    0x00000007169B9740,
    0x0000000C37F99209,
    0x000000031FF6DAB9,
    0x0000000C795190ED,
    0x0000000A7636E95F,
    0x00000009DF075841,
    0x000000055A083932,
    0x0000000A7CBDF630,
    0x0000000409EA4EF0,
    0x000000092A1991B6,
    0x00000004B078DEE9,
    0x0000000AE18CE9E4,
    0x00000005A6E1EF35,
    0x00000001A403BD59,
    0x000000031EA70A83,
    0x00000002BC3C4F3A,
    0x00000005C921B3CB,
    0x0000000042DA05C5,
    0x00000001F667D16B,
    0x0000000416A368CF,
    0x0000000FBC0A7A3B,
    0x00000009419F0C7C,
    0x000000081BE2FA03,
    0x000000034E2C172F,
    0x000000028648D8AE,
    0x0000000C7ACBB885,
    0x000000045F31EB6A,
    0x0000000D1CFC0A7B,
    0x000000042C4D260D,
    0x0000000CF6584097,
    0x000000094B132B14,
    0x00000003C5C5DF75,
    0x00000008AE596FEF,
    0x0000000AEA8054EB,
    0x00000000AE9CC573,
    0x0000000496FB731B,
    0x0000000EBF105662,
    0x0000000AF9C83A37,
    0x0000000C0D64CD6B,
    0x00000007B608159A,
    0x0000000E74431642,
    0x0000000D6FB9D900,
    0x0000000291E99DE0,
    0x000000010500BA9A,
    0x00000005CD05D037,
    0x0000000A87254FB2,
    0x00000009D7824A37,
    0x00000008B2C7B47C,
    0x000000030C788145,
    0x00000002F4E5A8BE,
    0x0000000BADB884DA,
    0x0000000026E0D5C9,
    0x00000006FDBAA32E,
    0x000000034758EB31,
    0x0000000565CD1B4F,
    0x00000002BFD90FB0,
    0x0000000093052A6B,
    0x0000000D3C13C4B9,
    0x00000002DAEA43BF,
    0x0000000A279762BC,
    0x0000000F1BD9F22C,
    0x00000004B7FEC94F,
    0x0000000545761D5A,
    0x00000007327DF411,
    0x00000001B52A442E,
    0x000000049B0CE108,
    0x000000024C764BC8,
    0x0000000374563045,
    0x0000000A3E8F91C6,
    0x00000000E6BD2241,
    0x0000000E0E52EE3C,
    0x000000007E8E3CAA,
    0x000000096C2B7372,
    0x000000033ACBDFDFA,
    0x0000000B15D91E54,
    0x0000000464759AC1,
    0x00000006886A1998,
    0x000000057F5D3958,
    0x00000005A1F5C1F5,
    0x00000000B58158AD,
    0x0000000E712053FB,
    0x00000005352DDB25,
    0x0000000414B98EA0,
    0x000000074F89F546,
    0x000000038A56B3C3,
    0x000000038DB0DC17,
    0x0000000AA016A755,
    0x0000000DC72366F5,
    0x00000000CEE93D75,
    0x0000000B2FE7A56B,
    0x0000000A847ED390,
    0x00000008713EF88C,
    0x0000000A217CC861,
    0x00000008BCA25D7B,
    0x0000000455526818,
    0x0000000EA3A7A180,
    0x0000000A9536E5E0,
    0x00000009B64A1975,
    0x00000005BFC756BC,
    0x0000000046AA169B,
    0x000000053A17F76F,
    0x00000004D6815274,
    0x0000000CCA9CF3F6,
    0x00000004013FCB8B,
    0x00000003D26CDFA5,
    0x00000005786231F7,
    0x00000007D4AB09AB,
    0x0000000960B5FFBC,
    0x00000008914DF0D4,
    0x00000002FC6F2213,
    0x0000000AC235637E,
    0x0000000151B28ED3,
    0x000000046F79B6DB,
    0x00000001382E0C9F,
    0x000000053ABF983A,
    0x0000000383C47ADE,
    0x00000003FCF88978,
    0x0000000EB9079DF7,
    0x000000009AF0714D,
    0x0000000DA19D1BB7,
    0x00000009A02749F8,
    0x00000001C62DAB9B,
    0x00000001A137E44B,
    0x00000002867718C7,
    0x000000035815525B,
    0x00000007CD35C550,
    0x00000002164F73A0,
    0x0000000E8B772FE0,
]

# All 30 codes for AprilTag 16h5 (from UMich source)
APRILTAG_16H5_CODES_SPIRAL = [
    0x00000000000027C8,
    0x00000000000031B6,
    0x0000000000003859,
    0x000000000000569C,
    0x0000000000006C76,
    0x0000000000007DDB,
    0x000000000000AF09,
    0x000000000000F5A1,
    0x000000000000FB8B,
    0x0000000000001CB9,
    0x00000000000028CA,
    0x000000000000E8DC,
    0x0000000000001426,
    0x0000000000005770,
    0x0000000000009253,
    0x000000000000B702,
    0x000000000000063A,
    0x0000000000008F34,
    0x000000000000B4C0,
    0x00000000000051EC,
    0x000000000000E6F0,
    0x0000000000005FA4,
    0x000000000000DD43,
    0x0000000000001AAA,
    0x000000000000E62F,
    0x0000000000006DBC,
    0x000000000000B6EB,
    0x000000000000DE10,
    0x000000000000154D,
    0x000000000000B57A,
]

# UMich bit ordering for 36h11 (spiral pattern)
# These are (x, y) coordinates for each bit position in UMich ordering
UMICH_36H11_BIT_ORDER = [
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (2, 2),
    (3, 2),
    (4, 2),
    (3, 3),
    (6, 1),
    (6, 2),
    (6, 3),
    (6, 4),
    (6, 5),
    (5, 2),
    (5, 3),
    (5, 4),
    (4, 3),
    (6, 6),
    (5, 6),
    (4, 6),
    (3, 6),
    (2, 6),
    (5, 5),
    (4, 5),
    (3, 5),
    (4, 4),
    (1, 6),
    (1, 5),
    (1, 4),
    (1, 3),
    (1, 2),
    (2, 5),
    (2, 4),
    (2, 3),
    (3, 4),
]

# UMich bit ordering for 16h5 (spiral pattern)
UMICH_16H5_BIT_ORDER = [
    (1, 1),
    (2, 1),
    (3, 1),
    (2, 2),
    (4, 1),
    (4, 2),
    (4, 3),
    (3, 2),
    (4, 4),
    (3, 4),
    (2, 4),
    (3, 3),
    (1, 4),
    (1, 3),
    (1, 2),
    (2, 3),
]

# ArUco 4x4_50 codes (extracted from OpenCV, already in row-major order)
# These do NOT need bit ordering conversion
ARUCO_4X4_50_CODES = [
    0x4CAD,
    0x59F0,
    0xB4CC,
    0x6299,
    0x792A,
    0xB39E,
    0x7479,
    0x4F23,
    0x5B7F,
    0x6AF3,
    0x899F,
    0xE588,
    0xED70,
    0xF054,
    0x8D24,
    0x7C64,
    0xA662,
    0x0066,
    0x7A36,
    0xF56E,
    0xD161,
    0xD40D,
    0xAB33,
    0x41BB,
    0xE27F,
    0x8E29,
    0x2735,
    0x2AA5,
    0xC484,
    0xF62C,
    0xA822,
    0x4DEA,
    0xF379,
    0xD30F,
    0x7510,
    0x9490,
    0xAE18,
    0xFF20,
    0x6FB0,
    0x5A38,
    0x18E8,
    0x1454,
    0x314C,
    0x4D1C,
    0x1724,
    0xD774,
    0xFCB4,
    0x26D2,
    0x740A,
    0xC80A,
]

# ArUco 4x4_100 codes (first 50 are identical to 4x4_50)
ARUCO_4X4_100_CODES = [
    0x4CAD,
    0x59F0,
    0xB4CC,
    0x6299,
    0x792A,
    0xB39E,
    0x7479,
    0x4F23,
    0x5B7F,
    0x6AF3,
    0x899F,
    0xE588,
    0xED70,
    0xF054,
    0x8D24,
    0x7C64,
    0xA662,
    0x0066,
    0x7A36,
    0xF56E,
    0xD161,
    0xD40D,
    0xAB33,
    0x41BB,
    0xE27F,
    0x8E29,
    0x2735,
    0x2AA5,
    0xC484,
    0xF62C,
    0xA822,
    0x4DEA,
    0xF379,
    0xD30F,
    0x7510,
    0x9490,
    0xAE18,
    0xFF20,
    0x6FB0,
    0x5A38,
    0x18E8,
    0x1454,
    0x314C,
    0x4D1C,
    0x1724,
    0xD774,
    0xFCB4,
    0x26D2,
    0x740A,
    0xC80A,
    0x298A,
    0x16AA,
    0x82BA,
    0xE9FA,
    0x8016,
    0xE616,
    0x2486,
    0x9786,
    0x48D6,
    0xA7F6,
    0xFBE6,
    0xD87E,
    0x0501,
    0x22C1,
    0x45D1,
    0x5EC9,
    0x3621,
    0x54A1,
    0x39A1,
    0x9139,
    0x85F9,
    0x3EDD,
    0x203D,
    0xDA6D,
    0x13FD,
    0xD5ED,
    0xF853,
    0x4693,
    0x1A9B,
    0xABCB,
    0x1933,
    0x05E3,
    0xECA3,
    0xBA97,
    0xA49F,
    0xDDDF,
    0x5477,
    0xB2EF,
    0xAEAC,
    0xB551,
    0xE86E,
    0xF350,
    0xD260,
    0x83B4,
    0x1B92,
    0x2FC2,
    0x6CF2,
    0xCBF2,
    0x2796,
    0xE30E,
]


def spiral_to_rowmajor(code: int, dim: int, bit_order: list) -> int:
    """Convert from UMich spiral bit ordering to row-major ordering.

    Args:
        code: The code in UMich spiral ordering
        dim: Grid dimension (4 for 16h5, 6 for 36h11)
        bit_order: List of (x, y) tuples for UMich bit positions

    Returns:
        Code in row-major ordering (bit 0 = top-left, row-major scan)
    """
    result = 0
    for spiral_bit_idx, (x, y) in enumerate(bit_order):
        # Check if this bit is set in the spiral-ordered code
        if code & (1 << spiral_bit_idx):
            # Convert 1-based (x, y) to 0-based row-major index
            row = y - 1
            col = x - 1
            rowmajor_idx = row * dim + col
            result |= 1 << rowmajor_idx
    return result


def generate_grid_points(dim: int, total_width: int) -> list:
    """Generate sample points for a dense grid centered in the tag.

    Args:
        dim: Dimension of data grid (e.g. 6 for 36h11)
        total_width: Total width of tag in modules (e.g. 8 for 36h11)

    Returns:
        List of (x, y) tuples in canonical coordinates [-1, 1]
    """
    points = []
    # Data is usually centered.
    # For 36h11 (dim 6, width 8), data is indices 1..6 (0-based 1..6).
    # Center of tag is 4.0.
    # Module i center is i + 0.5.
    # Coordinate = (i + 0.5 - total_width/2) * (2.0 / total_width)

    # Calculate offset to center the data grid
    # e.g. for dim=6, width=8, start_idx=1.
    start_idx = (total_width - dim) / 2

    for r in range(dim):
        for c in range(dim):
            y_idx = start_idx + r
            x_idx = start_idx + c

            y = (y_idx + 0.5 - total_width / 2.0) * (2.0 / total_width)
            x = (x_idx + 0.5 - total_width / 2.0) * (2.0 / total_width)
            points.append((x, y))
    return points


def generate_sparse_points(bit_order: list, total_width: int) -> list:
    """Generate sample points for sparse bit layout.

    Args:
        bit_order: List of (x, y) module coordinates
        total_width: Total width of tag in modules

    Returns:
        List of (x, y) tuples in canonical coordinates [-1, 1]
    """
    if not bit_order:
        return []

    # Heuristic: Center the bounding box of the points
    xs = [p[0] for p in bit_order]
    ys = [p[1] for p in bit_order]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # Force center to 0,0 if the range looks like it should be centered?
    # For 41h12 (-2..6), center is 2.
    # If we trust the file layout logic, assume explicit coordinates are correct relative to some origin.
    # But usually OpenCV/AprilTag define origin at center.
    # If standard tags use -2..6, likely origin is NOT center (or shifted).
    # Centering it ensures robustness.

    scale = 2.0 / total_width
    points = []
    for x_idx, y_idx in bit_order:
        # Note: UMich bit_y is usually row (down), so it maps to y.
        # But we need to check if Y needs inverting.
        # Canonical Y usually points DOWN in image coords?
        # Standard: X right, Y down. Canonical: -1,-1 is top-left.
        # So Y increases down.

        x = (x_idx - center_x) * scale
        y = (y_idx - center_y) * scale
        points.append((x, y))

    return points


def generate_point_array(name: str, points: list) -> str:
    """Generate Rust array for sample points."""
    lines = [f"/// Sample points for {name} (canonical coordinates)."]
    lines.append("#[rustfmt::skip]")
    lines.append("#[allow(clippy::unreadable_literal)]")
    lines.append(f"pub static {name}: [(f64, f64); {len(points)}] = [")

    for i in range(0, len(points), 2):
        chunk = points[i : i + 2]
        strs = [f"({p[0]:.6}, {p[1]:.6})" for p in chunk]
        lines.append(f"    {', '.join(strs)},")

    lines.append("];")
    return "\n".join(lines)


def generate_rust_array(name: str, codes: list, dim: int, bit_order: list) -> str:
    """Generate Rust array declaration for a dictionary."""
    # Convert all codes to row-major ordering
    converted = [spiral_to_rowmajor(c, dim, bit_order) for c in codes]

    lines = [f"pub static {name}: [u64; {len(converted)}] = ["]

    # Format 4 codes per line
    for i in range(0, len(converted), 4):
        chunk = converted[i : i + 4]
        # Format as 0x1234_5678_9abc
        hex_strs = []
        for c in chunk:
            s = f"{c:012x}"
            hex_strs.append(f"0x{s[:4]}_{s[4:8]}_{s[8:]}")

        lines.append(f"    {', '.join(hex_strs)},")

    lines.append("];")
    return "\n".join(lines)


def generate_rust_array_raw(name: str, codes: list) -> str:
    """Generate Rust array declaration for codes already in row-major order (no conversion)."""
    lines = [f"pub static {name}: [u64; {len(codes)}] = ["]

    # Format 4 codes per line
    for i in range(0, len(codes), 4):
        chunk = codes[i : i + 4]
        hex_strs = []
        for c in chunk:
            s = f"{c:012x}"
            hex_strs.append(f"0x{s[:4]}_{s[4:8]}_{s[8:]}")
        lines.append(f"    {', '.join(hex_strs)},")

    lines.append("];")
    return "\n".join(lines)


def generate_dictionaries_rs() -> str:
    # Pre-calculate points
    points_36h11 = generate_grid_points(6, 8)
    points_16h5 = generate_grid_points(4, 6)
    points_aruco = generate_grid_points(4, 6)

    parts = [
        "//! Tag family dictionaries.",
        "//!",
        "//! This module contains pre-generated code tables for AprilTag families.",
        "//! Codes are in row-major bit ordering for efficient extraction.",
        "",
        UMICH_LICENSE,
        "",
        "use std::borrow::Cow;",
        "use std::collections::HashMap;",
        "",
        "/// A tag family dictionary.",
        "#[derive(Clone, Debug)]",
        "pub struct TagDictionary {",
        "    /// Name of the tag family.",
        "    pub name: Cow<'static, str>,",
        "    /// Grid dimension (e.g., 6 for 36h11).",
        "    pub dimension: usize,",
        "    /// Minimum hamming distance of the family.",
        "    pub hamming_distance: usize,",
        "    /// Pre-computed sample points in canonical tag coordinates [-1, 1].",
        "    pub sample_points: Cow<'static, [(f64, f64)]>,",
        "    /// Raw code table.",
        "    codes: Cow<'static, [u64]>,",
        "    /// Lookup table for O(1) exact matching.",
        "    code_to_id: HashMap<u64, u16>,",
        "}",
        "",
        "impl TagDictionary {",
        "    /// Create a new dictionary from static code table.",
        "    #[must_use]",
        "    pub fn new(",
        "        name: &'static str,",
        "        dimension: usize,",
        "        hamming_distance: usize,",
        "        codes: &'static [u64],",
        "        sample_points: &'static [(f64, f64)],",
        "    ) -> Self {",
        "        let mut code_to_id = HashMap::with_capacity(codes.len());",
        "        for (id, &code) in codes.iter().enumerate() {",
        "            code_to_id.insert(code, id as u16);",
        "        }",
        "        Self {",
        "            name: Cow::Borrowed(name),",
        "            dimension,",
        "            hamming_distance,",
        "            sample_points: Cow::Borrowed(sample_points),",
        "            codes: Cow::Borrowed(codes),",
        "            code_to_id,",
        "        }",
        "    }",
        "",
        "    /// Create a new custom dictionary from a vector of codes.",
        "    #[must_use]",
        "    pub fn new_custom(",
        "        name: String,",
        "        dimension: usize,",
        "        hamming_distance: usize,",
        "        codes: Vec<u64>,",
        "        sample_points: Vec<(f64, f64)>,",
        "    ) -> Self {",
        "        let mut code_to_id = HashMap::with_capacity(codes.len());",
        "        for (id, &code) in codes.iter().enumerate() {",
        "            code_to_id.insert(code, id as u16);",
        "        }",
        "        Self {",
        "            name: Cow::Owned(name),",
        "            dimension,",
        "            hamming_distance,",
        "            sample_points: Cow::Owned(sample_points),",
        "            codes: Cow::Owned(codes),",
        "            code_to_id,",
        "        }",
        "    }",
        "",
        "    /// Get number of codes in dictionary.",
        "    #[must_use]",
        "    pub fn len(&self) -> usize {",
        "        self.codes.len()",
        "    }",
        "",
        "    /// Check if dictionary is empty.",
        "    #[must_use]",
        "    pub fn is_empty(&self) -> bool {",
        "        self.codes.is_empty()",
        "    }",
        "",
        "    /// Get the raw code for a given ID.",
        "    #[must_use]",
        "    pub fn get_code(&self, id: u16) -> Option<u64> {",
        "        self.codes.get(id as usize).copied()",
        "    }",
        "",
        "    /// Decode bits, trying all 4 rotations.",
        "    /// Returns (id, hamming_distance) if found within tolerance.",
        "    #[must_use]",
        "    pub fn decode(&self, bits: u64, max_hamming: u32) -> Option<(u16, u32)> {",
        "        let mask = if self.dimension * self.dimension <= 64 {",
        "            (1u64 << (self.dimension * self.dimension)) - 1",
        "        } else {",
        "            u64::MAX",
        "        };",
        "        let bits = bits & mask;",
        "",
        "        // Try exact match first (covers ~60% of clean reads)",
        "        let mut rbits = bits;",
        "        for _ in 0..4 {",
        "            if let Some(&id) = self.code_to_id.get(&rbits) {",
        "                return Some((id, 0));",
        "            }",
        "            if self.sample_points.len() == self.dimension * self.dimension {",
        "                 rbits = rotate90(rbits, self.dimension);",
        "            } else {",
        "                 break;",
        "            }",
        "        }",
        "",
        "        if max_hamming > 0 {",
        "            let mut best: Option<(u16, u32)> = None;",
        "            for (id, &code) in self.codes.iter().enumerate() {",
        "                let mut rbits = bits;",
        "                for _ in 0..4 {",
        "                    let hamming = (rbits ^ code).count_ones();",
        "                    if hamming <= max_hamming && best.is_none_or(|(_, h)| hamming < h) {",
        "                        best = Some((id as u16, hamming));",
        "                    }",
        "                    if self.sample_points.len() == self.dimension * self.dimension {",
        "                        rbits = rotate90(rbits, self.dimension);",
        "                    } else {",
        "                        break;",
        "                    }",
        "                }",
        "            }",
        "            return best;",
        "        }",
        "        None",
        "    }",
        "}",
        "",
        "/// Rotates a square bit pattern 90 degrees clockwise.",
        "#[must_use]",
        "pub fn rotate90(bits: u64, dim: usize) -> u64 {",
        "    let mut res = 0u64;",
        "    for y in 0..dim {",
        "        for x in 0..dim {",
        "            if (bits >> (y * dim + x)) & 1 != 0 {",
        "                let nx = dim - 1 - y;",
        "                let ny = x;",
        "                res |= 1 << (ny * dim + nx);",
        "            }",
        "        }",
        "    }",
        "    res",
        "}",
        "",
        "// ============================================================================",
        "// AprilTag 36h11 (587 codes)",
        "// ============================================================================",
        "",
        generate_point_array("APRILTAG_36H11_POINTS", points_36h11),
        "",
        "/// AprilTag 36h11 code table (587 entries, row-major bit ordering).",
        "#[rustfmt::skip]",
        generate_rust_array(
            "APRILTAG_36H11_CODES", APRILTAG_36H11_CODES_SPIRAL, 6, UMICH_36H11_BIT_ORDER
        ),
        "",
        "/// AprilTag 36h11 dictionary singleton.",
        "pub static APRILTAG_36H11: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
        '    TagDictionary::new("36h11", 6, 11, &APRILTAG_36H11_CODES, &APRILTAG_36H11_POINTS)',
        "});",
        "",
        "// ============================================================================",
        "// AprilTag 16h5 (30 codes)",
        "// ============================================================================",
        "",
        generate_point_array("APRILTAG_16H5_POINTS", points_16h5),
        "",
        "/// AprilTag 16h5 code table (30 entries, row-major bit ordering).",
        "#[rustfmt::skip]",
        generate_rust_array(
            "APRILTAG_16H5_CODES", APRILTAG_16H5_CODES_SPIRAL, 4, UMICH_16H5_BIT_ORDER
        ),
        "",
        "/// AprilTag 16h5 dictionary singleton.",
        "pub static APRILTAG_16H5: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
        '    TagDictionary::new("16h5", 4, 5, &APRILTAG_16H5_CODES, &APRILTAG_16H5_POINTS)',
        "});",
        "",
        "// ============================================================================",
        "// ArUco 4x4_50 (50 codes)",
        "// ============================================================================",
        "",
        generate_point_array("ARUCO_4X4_POINTS", points_aruco),
        "",
        "/// ArUco 4x4_50 code table (50 entries, row-major bit ordering).",
        "#[rustfmt::skip]",
        generate_rust_array_raw("ARUCO_4X4_50_CODES", ARUCO_4X4_50_CODES),
        "",
        "/// ArUco 4x4_50 dictionary singleton.",
        "pub static ARUCO_4X4_50: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
        '    TagDictionary::new("4X4_50", 4, 1, &ARUCO_4X4_50_CODES, &ARUCO_4X4_POINTS)',
        "});",
        "",
        "// ============================================================================",
        "// ArUco 4x4_100 (100 codes)",
        "// ============================================================================",
        "",
        "/// ArUco 4x4_100 code table (100 entries, row-major bit ordering).",
        "#[rustfmt::skip]",
        generate_rust_array_raw("ARUCO_4X4_100_CODES", ARUCO_4X4_100_CODES),
        "",
        "/// ArUco 4x4_100 dictionary singleton.",
        "pub static ARUCO_4X4_100: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
        '    TagDictionary::new("4X4_100", 4, 1, &ARUCO_4X4_100_CODES, &ARUCO_4X4_POINTS)',
        "});",
    ]

    if apriltag_41h12_data:
        points_41h12 = generate_sparse_points(apriltag_41h12_data.UMICH_41H12_BIT_ORDER, 9)
        parts.extend(
            [
                "",
                "// ============================================================================",
                "// AprilTag 41h12",
                "// ============================================================================",
                "",
                generate_point_array("APRILTAG_41H12_POINTS", points_41h12),
                "",
                "/// AprilTag 41h12 code table.",
                "#[rustfmt::skip]",
                generate_rust_array_raw(
                    "APRILTAG_41H12_CODES", apriltag_41h12_data.APRILTAG_41H12_CODES_SPIRAL
                ),
                "",
                "/// AprilTag 41h12 dictionary singleton.",
                "pub static APRILTAG_41H12: std::sync::LazyLock<TagDictionary> = std::sync::LazyLock::new(|| {",
                '    TagDictionary::new("41h12", 9, 12, &APRILTAG_41H12_CODES, &APRILTAG_41H12_POINTS)',
                "});",
            ]
        )

    parts.append("")
    return "\n".join(parts)


def main():
    script_dir = Path(__file__).parent
    output_path = script_dir.parent / "crates" / "locus-core" / "src" / "dictionaries.rs"

    content = generate_dictionaries_rs()
    output_path.write_text(content)
    print(f"Generated {output_path}")
    print(f"  - AprilTag 36h11: {len(APRILTAG_36H11_CODES_SPIRAL)} codes")
    print(f"  - AprilTag 16h5: {len(APRILTAG_16H5_CODES_SPIRAL)} codes")


if __name__ == "__main__":
    main()
