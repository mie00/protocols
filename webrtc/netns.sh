#!/usr/bin/env bash
set -e

function usage {
	echo "usage: $0 [create|remove] netns_name [network_ip]"
	exit 1
}

if [ "$#" -lt "1" -o "$1" = "-h" -o "$1" = "--help" ]; then
	usage
fi

if [ x"$USER" != x"root" ]; then
	echo "Executing as root"
	exec sudo $0 $@
fi

if [ x"$1" != x"create" -a x"$1" != x"remove" ]; then
	comm="create"
	netns=${1}
	ip=${2}
else
	comm=${1}
	netns=${2}
	ip=${3}
fi
export comm
export netns
export ip

if [ x"${comm}" = x"create" ]; then
	if [ x"$(ip netns | grep "^$netns " | wc -l)" = x"0" ]; then
		while [ x"${ip}" = x"" ]; do
			num=$(shuf -i 2-254 -n 1)
			export trial=10.0.$num.0
			if [ -z "$(ip link | grep \"^$trial\")" ]; then
				export ip=$trial
			fi
		done
		echo "creating a new namespace with ip ${ip}"
		ip link add ${netns}_bridge type bridge
		ip link add ${netns}_veth0 type veth peer name ${netns}_veth1
		ip netns add ${netns}
		ip link set ${netns}_bridge up
		ip link set ${netns}_veth0 up
		ip link set ${netns}_veth0 master ${netns}_bridge
		ip link set ${netns}_veth1 netns ${netns}
		ip address add ${ip}1/24 dev ${netns}_bridge
		ip netns exec ${netns} ip link set ${netns}_veth1 name eth0
		ip netns exec ${netns} ip link set eth0  up
		ip netns exec ${netns} ip address add ${ip}2/24 dev eth0
		ip netns exec ${netns} ip route add default via ${ip}1 dev eth0
		ip netns exec ${netns} echo 'nameserver 8.8.8.8' > /etc/resolv.conf
		iptables -A FORWARD -s ${ip}/24 ! -d ${ip}/24 -j ACCEPT
		iptables -A FORWARD ! -s ${ip}/24 -d ${ip}/24 -j ACCEPT
		iptables -t nat -A POSTROUTING -s ${ip}/24 ! -d ${ip}/24 -j MASQUERADE
	fi
	echo "attatching to network namespace ${netns}"
	ip netns exec ${netns} sudo -u ${SUDO_USER:-$USER} -i
else
	if [ x"$(ip netns | grep "^$netns " | wc -l)" = x"0" ]; then
		echo "error: ${netns} namespace doesn't exist"
		exit 1
	fi
	ip=$(ip route show dev ${netns}_bridge | cut -d '/' -f 1)
	iptables -t nat -D POSTROUTING -s ${ip}/24 ! -d ${ip}/24 -j MASQUERADE
	iptables -D FORWARD -s ${ip}/24 ! -d ${ip}/24 -j ACCEPT
	iptables -D FORWARD ! -s ${ip}/24 -d ${ip}/24 -j ACCEPT
	ip netns exec ${netns} ip link delete eth0
	ip netns delete ${netns}
	ip link delete ${netns}_bridge
fi
