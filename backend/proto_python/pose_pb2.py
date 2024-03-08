# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pose.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="pose.proto",
    package="",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\npose.proto"r\n\rPoseTimestamp\x12\x11\n\ttimestamp\x18\x01 \x01(\x01\x12\x17\n\x0fposeTranslation\x18\x02 \x03(\x02\x12\x11\n\trotMatrix\x18\x03 \x03(\x02\x12\x10\n\x08quatImag\x18\x04 \x03(\x02\x12\x10\n\x08quatReal\x18\x05 \x01(\x02"\xe1\x01\n\x08PoseData\x12,\n\x0cmappingPhase\x18\x01 \x01(\x0b\x32\x16.PoseData.MappingPhase\x12\x36\n\x11localizationPhase\x18\x02 \x01(\x0b\x32\x1b.PoseData.LocalizationPhase\x1a\x34\n\x0cMappingPhase\x12$\n\x0cmeasurements\x18\x01 \x03(\x0b\x32\x0e.PoseTimestamp\x1a\x39\n\x11LocalizationPhase\x12$\n\x0cmeasurements\x18\x01 \x03(\x0b\x32\x0e.PoseTimestampb\x06proto3',
)


_POSETIMESTAMP = _descriptor.Descriptor(
    name="PoseTimestamp",
    full_name="PoseTimestamp",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="timestamp",
            full_name="PoseTimestamp.timestamp",
            index=0,
            number=1,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="poseTranslation",
            full_name="PoseTimestamp.poseTranslation",
            index=1,
            number=2,
            type=2,
            cpp_type=6,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="rotMatrix",
            full_name="PoseTimestamp.rotMatrix",
            index=2,
            number=3,
            type=2,
            cpp_type=6,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="quatImag",
            full_name="PoseTimestamp.quatImag",
            index=3,
            number=4,
            type=2,
            cpp_type=6,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="quatReal",
            full_name="PoseTimestamp.quatReal",
            index=4,
            number=5,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=14,
    serialized_end=128,
)


_POSEDATA_MAPPINGPHASE = _descriptor.Descriptor(
    name="MappingPhase",
    full_name="PoseData.MappingPhase",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="measurements",
            full_name="PoseData.MappingPhase.measurements",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=245,
    serialized_end=297,
)

_POSEDATA_LOCALIZATIONPHASE = _descriptor.Descriptor(
    name="LocalizationPhase",
    full_name="PoseData.LocalizationPhase",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="measurements",
            full_name="PoseData.LocalizationPhase.measurements",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=299,
    serialized_end=356,
)

_POSEDATA = _descriptor.Descriptor(
    name="PoseData",
    full_name="PoseData",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="mappingPhase",
            full_name="PoseData.mappingPhase",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="localizationPhase",
            full_name="PoseData.localizationPhase",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[
        _POSEDATA_MAPPINGPHASE,
        _POSEDATA_LOCALIZATIONPHASE,
    ],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=131,
    serialized_end=356,
)

_POSEDATA_MAPPINGPHASE.fields_by_name["measurements"].message_type = _POSETIMESTAMP
_POSEDATA_MAPPINGPHASE.containing_type = _POSEDATA
_POSEDATA_LOCALIZATIONPHASE.fields_by_name["measurements"].message_type = _POSETIMESTAMP
_POSEDATA_LOCALIZATIONPHASE.containing_type = _POSEDATA
_POSEDATA.fields_by_name["mappingPhase"].message_type = _POSEDATA_MAPPINGPHASE
_POSEDATA.fields_by_name["localizationPhase"].message_type = _POSEDATA_LOCALIZATIONPHASE
DESCRIPTOR.message_types_by_name["PoseTimestamp"] = _POSETIMESTAMP
DESCRIPTOR.message_types_by_name["PoseData"] = _POSEDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PoseTimestamp = _reflection.GeneratedProtocolMessageType(
    "PoseTimestamp",
    (_message.Message,),
    {
        "DESCRIPTOR": _POSETIMESTAMP,
        "__module__": "pose_pb2",
        # @@protoc_insertion_point(class_scope:PoseTimestamp)
    },
)
_sym_db.RegisterMessage(PoseTimestamp)

PoseData = _reflection.GeneratedProtocolMessageType(
    "PoseData",
    (_message.Message,),
    {
        "MappingPhase": _reflection.GeneratedProtocolMessageType(
            "MappingPhase",
            (_message.Message,),
            {
                "DESCRIPTOR": _POSEDATA_MAPPINGPHASE,
                "__module__": "pose_pb2",
                # @@protoc_insertion_point(class_scope:PoseData.MappingPhase)
            },
        ),
        "LocalizationPhase": _reflection.GeneratedProtocolMessageType(
            "LocalizationPhase",
            (_message.Message,),
            {
                "DESCRIPTOR": _POSEDATA_LOCALIZATIONPHASE,
                "__module__": "pose_pb2",
                # @@protoc_insertion_point(class_scope:PoseData.LocalizationPhase)
            },
        ),
        "DESCRIPTOR": _POSEDATA,
        "__module__": "pose_pb2",
        # @@protoc_insertion_point(class_scope:PoseData)
    },
)
_sym_db.RegisterMessage(PoseData)
_sym_db.RegisterMessage(PoseData.MappingPhase)
_sym_db.RegisterMessage(PoseData.LocalizationPhase)


# @@protoc_insertion_point(module_scope)
