#Requires -RunAsAdministrator

Get-PnpDevice
$gpu = Get-PnpDevice -FriendlyName 'NVIDIA*'
nvidia-smi -i 0 -fdm 0
Disable-PnpDevice -InstanceId $gpu.InstanceId -Confirm:$false
Enable-PnpDevice -InstanceId $gpu.InstanceId -Confirm:$false
