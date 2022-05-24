import simpy 
from typing import List, Union, Any, Optional, Dict

class Mixin:

    def Purification(self, 
                              process: Dict, 
                              node1: Any, 
                              node2: Any, 
                              num_required: Optional[int] = 1, 
                              label_in: Optional[str] = 'Physical', 
                              label_out: Optional[str] = 'Purified',
                              protocol: Optional[str] = 'Ss-Dp',
                              note: Optional[Union[str, List]] = None):
        """This process will not induce any time delay, hence when `label_in` resources are available,
           it will fire an independent process for purification which perform actual protocol.

        Args:
            process (Dict): Dictionary of contain information of process.
            node1 (Any): node 1 which this process is process.
            node2 (Any): node 2 which this process is process.
            num_required (Optional[int], optional): Number of time that this process needed to looped. Defaults to 1.
            label_in (Optional[str], optional): Input label of resource. Defaults to 'Physical'.
            label_out (Optional[str], optional): Output label of resource. Defaults to 'Purified'.
            protocol (Optional[str], optional): Protocol used for entanglement purification. Defaults to 'Ss-Dp'.
            note (Optional[Union[str, List]], optional): Addition note for process. Defaults to None.

        Yields:
            _type_: _description_
        """

        # Valiate node order
        node1, node2 = self.validateNodeOrder(node1, node2)

        table = self.resourceTables['physicalResourceTable']

        while process['isSuccess'] < num_required:
            
            if protocol == 'Ss-Dp':
                # Request 3 Bell pairs

                if type(label_in) is str:
                    label_in = [label_in]*3
                
                event = yield simpy.AllOf(self.env, [table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in[0]), 
                                                     table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in[1]),
                                                     table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in[2])])

                Bells = []
                for i in range(3):
                    bell = yield event.events[i]
                    Bells.append(bell)


            elif protocol in ['X-purification', 'Z-purification']:

                if type(label_in) is str:
                    label_in = [label_in]*2

                event = yield simpy.AllOf(self.env, [table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in[0]), 
                                                     table[f'{node1}-{node2}'].get(lambda bell: bell[2] == label_in[1])])
                
                Bells = []
                for i in range(2):
                    bell = yield event.events[i]
                    Bells.append(bell)

            info = (protocol, Bells, node1, node2, table, label_out, num_required, process, note)
            self.env.process(self._independentPurification(info))


    def _independentPurification(self, info):

        protocol, Bells, node1, node2, table, label_out, num_required, process, note = info

        # Sort Bell pair in timely order
        if note == 'Sorted':
            Bells.sort(key=lambda pair: pair[0].initiateTime if pair[0].initiateTime <= pair[1].initiateTime else pair[1].initiateTime)
            

        if protocol == 'Ss-Dp':
            # Request 3 Bell pairs

            # Purify first pair with second and third
            # X purification
            Bells[1][0].CNOT_gate(Bells[0][0])
            Bells[1][1].CNOT_gate(Bells[0][1])

            # Z purification
            Bells[0][0].CNOT_gate(Bells[2][0])
            Bells[0][1].CNOT_gate(Bells[2][1])

            # measure
            x_result1, x_result2 = Bells[1][0].measureZ(), Bells[1][1].measureZ()
            z_result1, z_result2 = Bells[2][0].measureX(), Bells[2][1].measureX()

            # classical notification for result and also notify that Bell pairs is free now.
            yield self.env.process(self.classicalCommunication(node1, node2))

            if (x_result1 == x_result2) and (z_result1 == z_result2):
                # The result is agree, keep the first pair
                self.createLinkResource(node1, node2, Bells[0][0], Bells[0][1], table, label=label_out)
                for bell in Bells[1:]:
                    bell[0].setFree();bell[1].setFree()
                    self.QubitsTables[bell[0].table][bell[0].qnics_address][f'QNICs-{bell[0].qubit_node_address}'].put(bell[0])
                    self.QubitsTables[bell[1].table][bell[1].qnics_address][f'QNICs-{bell[1].qubit_node_address}'].put(bell[1])
                # self.updateLog({'Time': self.env.now, 'Message': f'Purification for {Bells[0][0].qubitID}-{Bells[0][1].qubitID} success'})
            else:
                # Fail discard all
                for bell in Bells:
                    bell[0].setFree();bell[1].setFree()
                    self.QubitsTables[bell[0].table][bell[0].qnics_address][f'QNICs-{bell[0].qubit_node_address}'].put(bell[0])
                    self.QubitsTables[bell[1].table][bell[1].qnics_address][f'QNICs-{bell[1].qubit_node_address}'].put(bell[1])
                # self.updateLog({'Time': self.env.now, 'Message': f'Purification for {Bells[0][0].qubitID}-{Bells[0][1].qubitID} fail'})

        

        elif protocol in ['X-purification', 'Z-purification']:

            # Purify first pair with second and third
            # X purification
            Bells[1][0].CNOT_gate(Bells[0][0])
            Bells[1][1].CNOT_gate(Bells[0][1])
            
            # measure
            if protocol == 'Z-purification':
                result1, result2 = Bells[1][0].measureX(), Bells[1][1].measureX()
            elif protocol == 'X-purification':
                result1, result2 = Bells[1][0].measureZ(), Bells[1][1].measureZ()
            else:
                raise ValueError('Invalid purification protocol')

            # classical notification for result
            yield self.env.process(self.classicalCommunication(node1, node2))

            if result1 == result2:
                # The result is agree, keep the first pair
                self.createLinkResource(node1, node2, Bells[0][0], Bells[0][1], table, label=label_out)
                for bell in Bells[1:]:
                    bell[0].setFree();bell[1].setFree()
                    self.QubitsTables[bell[0].table][bell[0].qnics_address][f'QNICs-{bell[0].qubit_node_address}'].put(bell[0])
                    self.QubitsTables[bell[1].table][bell[1].qnics_address][f'QNICs-{bell[1].qubit_node_address}'].put(bell[1])
                # self.updateLog({'Time': self.env.now, 'Message': f'Purification for {node1}-{node2} success'})
            else:
                # Fail discard all
                for bell in Bells:
                    bell[0].setFree();bell[1].setFree()
                    self.QubitsTables[bell[0].table][bell[0].qnics_address][f'QNICs-{bell[0].qubit_node_address}'].put(bell[0])
                    self.QubitsTables[bell[1].table][bell[1].qnics_address][f'QNICs-{bell[1].qubit_node_address}'].put(bell[1])
                # self.updateLog({'Time': self.env.now, 'Message': f'Purification for {node1}-{node2} fail'})


        if not isinstance(num_required, bool):
            process['isSuccess'] += 1 
