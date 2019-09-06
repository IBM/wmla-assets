## Introduction
to be completed & include picture

## Prerequisites
to be completed (docker installation)

## Install & Start LDAP server (provided as a docker image)
In this tutorial, the LDAP server is provided as a docker container.  There are many advantages to use an already built docker image for LDAP such as: 
*	Isolation (no physical software installation on the host)
*	Reproducibility (easy to reproduce in other environments)
*	Fast deployment (no need to spend time on installation and configuration)

### Create 2 directories that will contain LDAP persistent data
The LDAP data generated when you create a schema, a user or a group should be persistent on the host

`# mkdir -p /data/slapd/database`<br>
`# mkdir -p /data/slapd/config`

### Start the LDAP server with `docker run` command

`#docker run --name openldap --detach -p 389:389 
--volume /data/slapd/database:/var/lib/ldap 
--volume /data/slapd/config:/etc/ldap/slapd.d 
-e DOMAIN=mycluster.wmla -e PASSWORD=admin siji/openldap-ppc64le:2.4.42`


## Create LDIF File
Create a LDIF file containing LDAP entries for groups and users. The default password set in ldif file for users (LOB{x}_user{y}) is passw0rd. Passwords are SSHA encrypted and you can set your own passwords by copying the output of the following command:

`# slappasswd -h {SSHA} -s <password>`

If `slappasswd` utility is missing on your system, you can use the one provided by the LDAP docker image.

`# docker run --rm siji/openldap-ppc64le:2.4.42 slappasswd -h {SSHA} -s <password>`


~~~~
# Entry 1: ou=users,dc=mycluster,dc=wmla
dn: ou=users,dc=mycluster,dc=wmla
objectclass: organizationalUnit
objectclass: top
ou: users

# Entry 2: cn=lob1_user1,ou=users,dc=mycluster,dc=wmla
dn: cn=lob1_user1,ou=users,dc=mycluster,dc=wmla
cn: lob1_user1
gidnumber: 500
givenname: lob1_user1
homedirectory: /home/lob1user1
loginshell: /bin/bash
objectclass: inetOrgPerson
objectclass: posixAccount
objectclass: top
sn: lob1_user1
uid: lob1_user1
uidnumber: 1011
userpassword: {SSHA}AG1bsppJ3JNcCDwn+buh5jrSrtCVMHsK

# Entry 3: cn=lob1_user2,ou=users,dc=mycluster,dc=wmla
dn: cn=lob1_user2,ou=users,dc=mycluster,dc=wmla
cn: lob1_user2
gidnumber: 500
givenname: lob1_user2
homedirectory: /home/lob1user2
loginshell: /bin/bash
objectclass: inetOrgPerson
objectclass: posixAccount
objectclass: top
sn: lob1_user2
uid: lob1_user2
uidnumber: 1012
userpassword: {SSHA}AG1bsppJ3JNcCDwn+buh5jrSrtCVMHsK

# Entry 4: cn=lob2_user1,ou=users,dc=mycluster,dc=wmla
dn: cn=lob2_user1,ou=users,dc=mycluster,dc=wmla
cn: lob2_user1
gidnumber: 500
givenname: lob2_user1
homedirectory: /home/lob2user1
loginshell: /bin/bash
objectclass: inetOrgPerson
objectclass: posixAccount
objectclass: top
sn: lob2_user1
uid: lob2_user1
uidnumber: 1021
userpassword: {SSHA}AG1bsppJ3JNcCDwn+buh5jrSrtCVMHsK

# Entry 5: cn=lob2_user2,ou=users,dc=mycluster,dc=wmla
dn: cn=lob2_user2,ou=users,dc=mycluster,dc=wmla
cn: lob2_user2
gidnumber: 500
givenname: lob2_user2
homedirectory: /home/lob2user2
loginshell: /bin/bash
objectclass: inetOrgPerson
objectclass: posixAccount
objectclass: top
sn: lob2_user2
uid: lob2_user2
uidnumber: 1022
userpassword: {SSHA}AG1bsppJ3JNcCDwn+buh5jrSrtCVMHsK


# Entry 6: cn=lob1_admin,ou=users,dc=mycluster,dc=wmla
dn: cn=lob1_admin,ou=users,dc=mycluster,dc=wmla
cn: lob1_admin
gidnumber: 500
givenname: lob1_admin
homedirectory: /home/lob1admin
loginshell: /bin/bash
objectclass: inetOrgPerson
objectclass: posixAccount
objectclass: top
sn: lob1_admin
uid: lob1_admin
uidnumber: 1050
userpassword: {SSHA}AG1bsppJ3JNcCDwn+buh5jrSrtCVMHsK

# Entry 7: cn=lob2_admin,ou=users,dc=mycluster,dc=wmla
dn: cn=lob2_admin,ou=users,dc=mycluster,dc=wmla
cn: lob2_admin
gidnumber: 500
givenname: lob2_admin
homedirectory: /home/lob2admin
loginshell: /bin/bash
objectclass: inetOrgPerson
objectclass: posixAccount
objectclass: top
sn: lob2_admin
uid: lob2_admin
uidnumber: 1051
userpassword: {SSHA}AG1bsppJ3JNcCDwn+buh5jrSrtCVMHsK
 	
~~~~
Once you file is ok, import it in LDAP server with the following command:

`# ldapadd -c -x -W -D "cn=admin,dc=mycluster,dc=wmla" -f file.ldif`

To verify the server is running well and LDAP entries were correctly imported, run 'ldapsearch' to check:

`# ldapsearch -x -h 10.3.64.216 -b 'dc=mycluster,dc=wmla' '(objectclass=*)'|grep dn|grep wmla`
## Enable LDAP user authentication on all the wmla cluster nodes. 

Repeat the steps below for all the nodes (master and compute) of your WMLA cluster

`# yum -y install openldap-clients nss-pam-ldapd`<br>
`# authconfig --enableldap --enableldapauth --enablemd5 --ldapserver=10.3.64.216 --ldapbasedn="dc=mycluster,dc=wmla" --enablemkhomedir –update`<br>
`# echo "session     optional      pam_mkhomedir.so" >> /etc/pam.d/password-auth` 


## Verify LDAP client is enabled on the cluster nodes

Check if you can list LDAP users from any nodes of the WMLA cluster LDAP users should be in the user list. If not going to the troubleshooting section<br>
For example:<br>
`# getent passwd | grep LOB`<br>
~~~~
LOB1_user1:*:1011:500:LOB1_user1:/home/lob1user1:/bin/bash
LOB1_user2:*:1012:500:LOB1_user2:/home/lob1user2:/bin/bash
LOB2_user1:*:1021:500:LOB2_user1:/home/lob2user1:/bin/bash
LOB2_user2:*:1022:500:LOB2_user2:/home/lob2user2:/bin/bash
~~~~

Check also remote access on different nodes of the cluster with LDAP users

`ssh host -l LOB1_user1`

If you did not change encrypted password (SSHA) in the ldif file, the password is passw0rd.
Upon the user's first logon, the user's home directory should be automatically created if not exist.

`Install WMLA cluster`
If not already done, install wmla cluster and don't start the cluster. If WMLA cluster is already installed, ensure the cluster is down before proceed to next steps.

## Enable LDAP external user authentication in WMLA

1) Modify the $EGO_CONFDIR/ego.conf, with the changes like below:
Assume your cluster installation top is /opt/ibm/spectrumcomputing. If HA is enabled, use the shared top directory. 
Change:
~~~~
EGO_SEC_PLUGIN=sec_ego_default
EGO_SEC_CONF=/opt/ibm/spectrumcomputing/kernel/conf
~~~~
To:
~~~~
EGO_SEC_PLUGIN=sec_ego_pam_default
EGO_SEC_CONF=/opt/ibm/spectrumcomputing/kernel/conf,0,INFO,/opt/ibm/spectrumcomputing/kernel/log
~~~~

2) Start up the cluster as usual (run `egosh ego start` on all the wmla hosts)
3) Check if the users are correctly retrieved by wmla with command

`# egosh user list`

~~~~
Admin
egoadmin
Guest
lob1_admin     lob1_admin
lob1_user1     lob1_user1
lob1_user2     lob1_user2
lob2_admin     lob2_admin
lob2_user1     lob2_user1
lob2_user2     lob2_user2
~~~~

