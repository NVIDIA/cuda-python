#Requires -RunAsAdministrator

Get-PnpDevice
$gpu = Get-PnpDevice -FriendlyName 'NVIDIA*'
nvidia-smi -i 0 -dm WDDM
Disable-PnpDevice -InstanceId $gpu.InstanceId -Confirm:$false
Enable-PnpDevice -InstanceId $gpu.InstanceId -Confirm:$false
