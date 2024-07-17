import threading
import time
import unittest
from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc_testing

import nillion_fl.fl_net.fl_service_pb2 as fl_pb2
import nillion_fl.fl_net.fl_service_pb2_grpc as fl_pb2_grpc
from nillion_fl.server import \
    FederatedLearningServicer  # Replace 'your_module' with the actual module name


class TestFederatedLearningServicer(unittest.TestCase):
    def setUp(self):
        self.servicer = FederatedLearningServicer(num_parties=2)
        self.test_server = grpc_testing.server_from_dictionary(
            {
                fl_pb2.DESCRIPTOR.services_by_name[
                    "FederatedLearningService"
                ]: self.servicer
            },
            grpc_testing.strict_real_time(),
        )

    # def test_register_client(self):
    #     print("." * 100)
    #     print("Testing test_register_client")
    #     print("." * 100)
    #     request = fl_pb2.RegisterRequest()
    #     method = self.test_server.invoke_unary_unary(
    #         fl_pb2.DESCRIPTOR.services_by_name['FederatedLearningService'].methods_by_name['RegisterClient'],
    #         (),
    #         request,
    #         None
    #     )
    #     response, _, _, _ = method.termination()
    #     self.assertIsInstance(response, fl_pb2.ClientInfo)
    #     self.assertNotEqual(response.client_id, -1)
    #     self.assertNotEqual(response.token, "")

    # def test_register_client_max_reached(self):
    #     print("." * 100)
    #     print("Testing test_register_client_max_reached")
    #     print("." * 100)
    #     # Register two clients (max number)
    #     for _ in range(2):
    #         request = fl_pb2.RegisterRequest()
    #         method = self.test_server.invoke_unary_unary(
    #             fl_pb2.DESCRIPTOR.services_by_name['FederatedLearningService'].methods_by_name['RegisterClient'],
    #             (),
    #             request,
    #             None
    #         )
    #         method.termination()

    #     # Try to register a third client
    #     request = fl_pb2.RegisterRequest()
    #     method = self.test_server.invoke_unary_unary(
    #         fl_pb2.DESCRIPTOR.services_by_name['FederatedLearningService'].methods_by_name['RegisterClient'],
    #         (),
    #         request,
    #         None
    #     )
    #     response, _, _, _ = method.termination()
    #     self.assertEqual(response.client_id, -1)
    #     self.assertEqual(response.token, "")

    # @patch('threading.Thread')
    # def test_schedule_learning_iteration(self, mock_thread):
    #     token = self.servicer.RegisterClient(fl_pb2.RegisterRequest(), None).token

    #     def mock_request_iterator():
    #         yield fl_pb2.StoreIDs(token=token, store_ids=["id1", "id2"])

    #     context = MagicMock()
    #     responses = list(self.servicer.ScheduleLearningIteration(mock_request_iterator(), context))

    #     self.assertTrue(mock_thread.called)
    #     self.assertEqual(len(responses), 1)
    #     self.assertIsInstance(responses[0], fl_pb2.ScheduleRequest)

    # def test_handle_client_disconnect(self):
    #     token = self.servicer.RegisterClient(fl_pb2.RegisterRequest(), None).token
    #     stream_id = list(self.servicer.active_streams.keys())[0]

    #     self.servicer.handle_client_disconnect(stream_id)

    #     self.assertNotIn(token, self.servicer.clients)
    #     self.assertNotIn(stream_id, self.servicer.active_streams)
    #     self.assertNotIn(stream_id, self.servicer.client_threads)
    #     self.assertNotIn(stream_id, self.servicer.clients_queue)

    def test_schedule_learning(self):
        print("." * 100)
        print("Testing test_schedule_learning")
        print("." * 100)
        tokens = []
        for _ in range(2):
            request = fl_pb2.RegisterRequest()
            method = self.test_server.invoke_unary_unary(
                fl_pb2.DESCRIPTOR.services_by_name[
                    "FederatedLearningService"
                ].methods_by_name["RegisterClient"],
                (),
                request,
                None,
            )
            token = tokens.append(method.termination()[0].token)

        client_request_1 = fl_pb2.StoreIDs(token=tokens[0], store_ids=[])
        client_request_2 = fl_pb2.StoreIDs(token=tokens[1], store_ids=[])

        rpc_a = self.test_server.invoke_stream_stream(
            fl_pb2.DESCRIPTOR.services_by_name[
                "FederatedLearningService"
            ].methods_by_name["ScheduleLearningIteration"],
            (),
            None,
        )

        rpc_b = self.test_server.invoke_stream_stream(
            fl_pb2.DESCRIPTOR.services_by_name[
                "FederatedLearningService"
            ].methods_by_name["ScheduleLearningIteration"],
            (),
            None,
        )

        rpc_a.send_request(client_request_1)
        rpc_b.send_request(client_request_2)

        server_response_1 = rpc_a.take_response()
        server_response_2 = rpc_b.take_response()

        self.assertIsInstance(server_response_1, fl_pb2.ScheduleRequest)
        self.assertIsInstance(server_response_2, fl_pb2.ScheduleRequest)

        # self.assertEquals(server_response_1.program_id, server_response_2.program_id)
        self.assertEqual(server_response_1.user_id, server_response_2.user_id)
        self.assertEqual(server_response_1.batch_size, server_response_2.batch_size)
        self.assertEqual(server_response_1.num_parties, server_response_2.num_parties)

        client_request_1 = fl_pb2.StoreIDs(token=tokens[0], store_ids=["abc", "def"])
        client_request_2 = fl_pb2.StoreIDs(token=tokens[1], store_ids=["ghi", "jkl"])

        rpc_a.send_request(client_request_1)
        rpc_b.send_request(client_request_2)

        server_response_1 = rpc_a.take_response()
        server_response_2 = rpc_b.take_response()

        self.assertIsInstance(server_response_1, fl_pb2.ScheduleRequest)
        self.assertIsInstance(server_response_2, fl_pb2.ScheduleRequest)

        # self.assertEquals(server_response_1.program_id, server_response_2.program_id)
        self.assertEqual(server_response_1.user_id, server_response_2.user_id)
        self.assertEqual(server_response_1.batch_size, server_response_2.batch_size)
        self.assertEqual(server_response_1.num_parties, server_response_2.num_parties)

        self.servicer.learning_in_progress = False  # Stop learning

        rpc_a.send_request(
            client_request_1
        )  # Whenever it tries connecting, it is stopped
        rpc_b.send_request(
            client_request_2
        )  # Whenever it tries connecting, it is stopped

    # @patch('threading.Event.wait')
    # def test_end_learning_for_all_clients(self, mock_wait):
    #     mock_wait.return_value = True
    #     self.servicer.RegisterClient(fl_pb2.RegisterRequest(), None)
    #     self.servicer.RegisterClient(fl_pb2.RegisterRequest(), None)

    #     self.servicer.learning_in_progress = True
    #     self.servicer.end_learning_for_all_clients()

    #     self.assertFalse(self.servicer.learning_in_progress)
    #     for queue in self.servicer.clients_queue.values():
    #         self.assertFalse(queue.empty())
    #         message = queue.get()
    #         self.assertEqual(message.program_id, "-1")

    # @patch('grpc.server')
    # @patch('time.sleep', side_effect=KeyboardInterrupt)
    # def test_serve(self, mock_sleep, mock_server):
    #     mock_server_instance = MagicMock()
    #     mock_server.return_value = mock_server_instance

    #     with self.assertRaises(KeyboardInterrupt):
    #         self.servicer.serve()

    #     mock_server_instance.start.assert_called_once()
    #     mock_server_instance.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
