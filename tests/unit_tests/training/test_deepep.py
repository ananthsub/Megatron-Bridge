# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

from megatron.core.transformer import TransformerConfig

from megatron.bridge.training.deepep import DeepEPConfig, apply_deepep


class TestApplyDeepEP:
    """Test the apply_deepep function."""

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_ampere_gpu(self, mock_get_device_properties):
        """Test that DeepEP configs are applied on Ampere GPUs (SM80)."""
        # Mock Ampere GPU (compute capability 8.x)
        mock_properties = MagicMock()
        mock_properties.major = 8
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig
        config = MagicMock(spec=TransformerConfig)

        # Apply DeepEP
        apply_deepep(config)

        # Verify the correct configs were set
        assert config.moe_token_dispatcher_type == "flex"
        assert config.moe_enable_deepep is True
        assert config.moe_shared_expert_overlap is False

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_hopper_gpu(self, mock_get_device_properties):
        """Test that DeepEP configs are applied on Hopper GPUs (SM90)."""
        # Mock Hopper GPU (compute capability 9.x)
        mock_properties = MagicMock()
        mock_properties.major = 9
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig
        config = MagicMock(spec=TransformerConfig)

        # Apply DeepEP
        apply_deepep(config)

        # Verify the correct configs were set
        assert config.moe_token_dispatcher_type == "flex"
        assert config.moe_enable_deepep is True
        assert config.moe_shared_expert_overlap is False

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_older_gpu(self, mock_get_device_properties):
        """Test that DeepEP configs are not applied on older GPUs (e.g., Volta SM70)."""
        # Mock Volta GPU (compute capability 7.x)
        mock_properties = MagicMock()
        mock_properties.major = 7
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with initial values
        config = MagicMock(spec=TransformerConfig)
        initial_dispatcher = "default"
        initial_deepep = False
        initial_overlap = True

        config.moe_token_dispatcher_type = initial_dispatcher
        config.moe_enable_deepep = initial_deepep
        config.moe_shared_expert_overlap = initial_overlap

        # Apply DeepEP (should not modify config)
        apply_deepep(config)

        # Verify the configs remain unchanged
        assert config.moe_token_dispatcher_type == initial_dispatcher
        assert config.moe_enable_deepep == initial_deepep
        assert config.moe_shared_expert_overlap == initial_overlap

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_unsupported_gpu(self, mock_get_device_properties):
        """Test that DeepEP configs are not applied on unsupported GPUs (e.g., compute capability 6.x)."""
        # Mock Pascal GPU (compute capability 6.x)
        mock_properties = MagicMock()
        mock_properties.major = 6
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with initial values
        config = MagicMock(spec=TransformerConfig)
        initial_dispatcher = "legacy"
        initial_deepep = False
        initial_overlap = True

        config.moe_token_dispatcher_type = initial_dispatcher
        config.moe_enable_deepep = initial_deepep
        config.moe_shared_expert_overlap = initial_overlap

        # Apply DeepEP (should not modify config)
        apply_deepep(config)

        # Verify the configs remain unchanged
        assert config.moe_token_dispatcher_type == initial_dispatcher
        assert config.moe_enable_deepep == initial_deepep
        assert config.moe_shared_expert_overlap == initial_overlap

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_future_gpu(self, mock_get_device_properties):
        """Test that DeepEP configs are not applied on future unsupported GPUs (e.g., compute capability 10.x)."""
        # Mock future GPU (compute capability 10.x)
        mock_properties = MagicMock()
        mock_properties.major = 10
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with initial values
        config = MagicMock(spec=TransformerConfig)
        initial_dispatcher = "new"
        initial_deepep = False
        initial_overlap = True

        config.moe_token_dispatcher_type = initial_dispatcher
        config.moe_enable_deepep = initial_deepep
        config.moe_shared_expert_overlap = initial_overlap

        # Apply DeepEP (should not modify config)
        apply_deepep(config)

        # Verify the configs remain unchanged
        assert config.moe_token_dispatcher_type == initial_dispatcher
        assert config.moe_enable_deepep == initial_deepep
        assert config.moe_shared_expert_overlap == initial_overlap


class TestDeepEPConfig:
    """Test the DeepEPConfig class."""

    @patch("torch.cuda.get_device_properties")
    @patch("megatron.bridge.training.deepep.apply_deepep")
    def test_deepep_config_setup_calls_apply_deepep(self, mock_apply_deepep, mock_get_device_properties):
        """Test that DeepEPConfig.setup calls apply_deepep with the correct config."""
        # Mock Ampere GPU to ensure apply_deepep is called
        mock_properties = MagicMock()
        mock_properties.major = 8
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig
        config = MagicMock(spec=TransformerConfig)

        # Create and use DeepEPConfig
        deepep_config = DeepEPConfig()
        deepep_config.setup(config)

        # Verify that apply_deepep was called with the correct config
        mock_apply_deepep.assert_called_once_with(config)

    @patch("torch.cuda.get_device_properties")
    def test_deepep_config_integration_ampere(self, mock_get_device_properties):
        """Integration test: Test full flow with DeepEPConfig on Ampere GPU."""
        # Mock Ampere GPU
        mock_properties = MagicMock()
        mock_properties.major = 8
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig
        config = MagicMock(spec=TransformerConfig)

        # Create and use DeepEPConfig
        deepep_config = DeepEPConfig()
        deepep_config.setup(config)

        # Verify the correct configs were set
        assert config.moe_token_dispatcher_type == "flex"
        assert config.moe_enable_deepep is True
        assert config.moe_shared_expert_overlap is False

    @patch("torch.cuda.get_device_properties")
    def test_deepep_config_integration_unsupported(self, mock_get_device_properties):
        """Integration test: Test full flow with DeepEPConfig on unsupported GPU."""
        # Mock Volta GPU
        mock_properties = MagicMock()
        mock_properties.major = 7
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig with initial values
        config = MagicMock(spec=TransformerConfig)
        initial_dispatcher = "default"
        initial_deepep = False
        initial_overlap = True

        config.moe_token_dispatcher_type = initial_dispatcher
        config.moe_enable_deepep = initial_deepep
        config.moe_shared_expert_overlap = initial_overlap

        # Create and use DeepEPConfig
        deepep_config = DeepEPConfig()
        deepep_config.setup(config)

        # Verify the configs remain unchanged
        assert config.moe_token_dispatcher_type == initial_dispatcher
        assert config.moe_enable_deepep == initial_deepep
        assert config.moe_shared_expert_overlap == initial_overlap


class TestDeepEPConfigMisc:
    """Test miscellaneous aspects of DeepEP functionality."""

    @patch("torch.cuda.get_device_properties")
    def test_apply_deepep_device_properties_called_correctly(self, mock_get_device_properties):
        """Test that torch.cuda.get_device_properties is called with device 0."""
        # Mock Ampere GPU
        mock_properties = MagicMock()
        mock_properties.major = 8
        mock_get_device_properties.return_value = mock_properties

        # Create a mock TransformerConfig
        config = MagicMock(spec=TransformerConfig)

        # Apply DeepEP
        apply_deepep(config)

        # Verify get_device_properties was called with device 0
        mock_get_device_properties.assert_called_once_with(0)
