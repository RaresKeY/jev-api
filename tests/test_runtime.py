import subprocess
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock,patch
from decision_lab.runtime import ManagedServer


class RuntimeOwnership(unittest.TestCase):
    def test_missing_external_manager_fails_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            with patch('decision_lab.runtime.MANAGER', Path(directory)/'missing-manager'):
                with self.assertRaisesRegex(RuntimeError, 'standalone-dependencies'):
                    with ManagedServer('model.gguf', output, port=18080):
                        self.fail('Missing dependencies must stop before a launch')
            self.assertFalse(output.exists())

    def test_each_launch_has_its_own_name(self):
        first=ManagedServer('model.gguf','unused',port=18080)
        second=ManagedServer('model.gguf','unused',port=18080)
        self.assertNotEqual(first.name,second.name)

    def test_container_stop_precedes_manager_termination(self):
        server=ManagedServer('model.gguf','unused',port=18080)
        server.env={'LLM_CONTAINER':server.name}
        server.process=Mock()
        server.process.pid=123
        server.process.poll.return_value=None
        server.process.wait.side_effect=[subprocess.TimeoutExpired('manager',5),None]
        events=[]
        with patch('decision_lab.runtime.subprocess.run',side_effect=lambda *a,**k:events.append('stop')),patch('decision_lab.runtime.os.killpg',side_effect=lambda *a:events.append('terminate')):
            server.__exit__(None,None,None)
        self.assertEqual(events,['stop','terminate','stop'])

if __name__=='__main__':unittest.main()
