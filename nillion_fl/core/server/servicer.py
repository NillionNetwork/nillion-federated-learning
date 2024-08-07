# import grpc
# from concurrent import futures
# import time

# import nillion_fl.network.fl_service_pb2_grpc as fl_pb2_grpc
# from client_manager import ClientManager
# from learning_manager import LearningManager
# from nillion_integration import NillionIntegration
# from nillion_fl.logs import logger, uuid_str

# class FederatedLearningServicer(fl_pb2_grpc.FederatedLearningServiceServicer):
#     def __init__(self, host: str = "localhost", port: int = 50051, num_parties = 2, program_number=10, batch_size=1000):
#         self.host = host
#         self.port = port

#         self.client_manager = ClientManager(num_parties)
#         self.learning_manager = LearningManager(num_parties, batch_size)
#         self.nillion_integration = NillionIntegration(num_parties, program_number, batch_size)

#     def RegisterClient(self, request, context):
#         return self.client_manager.register_client(request, context)

#     def ScheduleLearningIteration(self, request_iterator, context):
#         return self.learning_manager.schedule_learning_iteration(request_iterator, context, self.client_manager, self.nillion_integration)

#     def serve(self):
#         server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#         fl_pb2_grpc.add_FederatedLearningServiceServicer_to_server(self, server)
#         server.add_insecure_port(f"{self.host}:{self.port}")

#         server.start()
#         logger.debug("Server started. Listening on port 50051.")
#         try:
#             while True:
#                 time.sleep(86400)
#         except KeyboardInterrupt:
#             logger.debug("Server stopping...")
#             self.learning_manager.stop()
#             logger.debug("All client threads stopped")
#             server.stop(1)
