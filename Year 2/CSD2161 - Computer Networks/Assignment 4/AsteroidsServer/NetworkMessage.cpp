/******************************************************************************/
/*!
\file   NetworkMessage.cpp
\brief  Network message serialization/deserialization
*/
/******************************************************************************/
#include "NetworkMessage.h"
#include <cstring>

// XOR each byte (zero-extended to uint16_t) of the first `len` bytes
uint16_t ComputeChecksum(const uint8_t* data, size_t len)
{
    uint16_t cs = 0;
    for (size_t i = 0; i < len; ++i)
        cs ^= static_cast<uint16_t>(data[i]);
    return cs;
}

bool BuildPacket(const NetworkMessage& msg, std::vector<uint8_t>& outBuffer)
{
    MsgHeader hdr      = msg.header;
    hdr.payloadLen     = static_cast<uint16_t>(msg.payload.size());
    hdr.checksum       = 0;

    // Compute checksum over the first 6 bytes of the header
    const uint8_t* hdrBytes = reinterpret_cast<const uint8_t*>(&hdr);
    hdr.checksum = ComputeChecksum(hdrBytes, 6);

    const size_t totalSize = sizeof(MsgHeader) + msg.payload.size();
    outBuffer.resize(totalSize);
    memcpy(outBuffer.data(), &hdr, sizeof(MsgHeader));
    if (!msg.payload.empty())
        memcpy(outBuffer.data() + sizeof(MsgHeader), msg.payload.data(), msg.payload.size());

    return true;
}

bool ParsePacket(const uint8_t* data, size_t len, NetworkMessage& outMsg)
{
    if (len < sizeof(MsgHeader))
        return false;

    MsgHeader hdr;
    memcpy(&hdr, data, sizeof(MsgHeader));

    // Validate checksum (bytes 0-5 of the received header, ignoring the checksum field itself)
    uint16_t cs = ComputeChecksum(data, 6);
    if (cs != hdr.checksum)
        return false;

    if (len < sizeof(MsgHeader) + hdr.payloadLen)
        return false;

    outMsg.header = hdr;
    outMsg.payload.assign(data + sizeof(MsgHeader),
                          data + sizeof(MsgHeader) + hdr.payloadLen);
    return true;
}
