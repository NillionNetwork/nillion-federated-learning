# -*- coding: utf-8 -*-
# pylint: skip-file
# mypy: ignore-errors
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nillion_fl/network/fl_service.proto
# Protobuf Python Version: 4.25.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n#nillion_fl/network/fl_service.proto\x12\x06\x66l_net"%\n\x0fRegisterRequest\x12\x12\n\nmodel_size\x18\x01 \x01(\x05"C\n\nClientInfo\x12\x11\n\tclient_id\x18\x01 \x01(\x05\x12\r\n\x05token\x18\x02 \x01(\t\x12\x13\n\x0bnum_parties\x18\x03 \x01(\x05"O\n\x08StoreIDs\x12\x10\n\x08store_id\x18\x01 \x01(\t\x12\x10\n\x08party_id\x18\x02 \x01(\t\x12\r\n\x05token\x18\x03 \x01(\t\x12\x10\n\x08\x62\x61tch_id\x18\x04 \x01(\x05"_\n\x0fScheduleRequest\x12\x12\n\nprogram_id\x18\x01 \x01(\t\x12\x0f\n\x07user_id\x18\x02 \x01(\t\x12\x12\n\nbatch_size\x18\x03 \x01(\x05\x12\x13\n\x0bnum_parties\x18\x04 \x01(\x05\x32\xa9\x01\n\x18\x46\x65\x64\x65ratedLearningService\x12?\n\x0eRegisterClient\x12\x17.fl_net.RegisterRequest\x1a\x12.fl_net.ClientInfo"\x00\x12L\n\x19ScheduleLearningIteration\x12\x10.fl_net.StoreIDs\x1a\x17.fl_net.ScheduleRequest"\x00(\x01\x30\x01\x62\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, "nillion_fl.network.fl_service_pb2", _globals
)
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    _globals["_REGISTERREQUEST"]._serialized_start = 47
    _globals["_REGISTERREQUEST"]._serialized_end = 84
    _globals["_CLIENTINFO"]._serialized_start = 86
    _globals["_CLIENTINFO"]._serialized_end = 153
    _globals["_STOREIDS"]._serialized_start = 155
    _globals["_STOREIDS"]._serialized_end = 234
    _globals["_SCHEDULEREQUEST"]._serialized_start = 236
    _globals["_SCHEDULEREQUEST"]._serialized_end = 331
    _globals["_FEDERATEDLEARNINGSERVICE"]._serialized_start = 334
    _globals["_FEDERATEDLEARNINGSERVICE"]._serialized_end = 503
# @@protoc_insertion_point(module_scope)
