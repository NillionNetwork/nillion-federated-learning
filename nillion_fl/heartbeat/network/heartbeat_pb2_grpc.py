# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from nillion_fl.heartbeat.network import (
    heartbeat_pb2 as nillion__fl_dot_heartbeat_dot_network_dot_heartbeat__pb2,
)


class HeartbeatServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SubscribeToHeartbeat = channel.stream_stream(
            "/HeartbeatService/SubscribeToHeartbeat",
            request_serializer=nillion__fl_dot_heartbeat_dot_network_dot_heartbeat__pb2.Heartbeat.SerializeToString,
            response_deserializer=nillion__fl_dot_heartbeat_dot_network_dot_heartbeat__pb2.Heartbeat.FromString,
        )


class HeartbeatServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SubscribeToHeartbeat(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_HeartbeatServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "SubscribeToHeartbeat": grpc.stream_stream_rpc_method_handler(
            servicer.SubscribeToHeartbeat,
            request_deserializer=nillion__fl_dot_heartbeat_dot_network_dot_heartbeat__pb2.Heartbeat.FromString,
            response_serializer=nillion__fl_dot_heartbeat_dot_network_dot_heartbeat__pb2.Heartbeat.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "HeartbeatService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class HeartbeatService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SubscribeToHeartbeat(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            "/HeartbeatService/SubscribeToHeartbeat",
            nillion__fl_dot_heartbeat_dot_network_dot_heartbeat__pb2.Heartbeat.SerializeToString,
            nillion__fl_dot_heartbeat_dot_network_dot_heartbeat__pb2.Heartbeat.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
