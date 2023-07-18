#!/usr/bin/env python
import sys
import math
import string
import numpy as np
import os
CUTOFF = 25

def torsion_angle(i, j, k, l):
    RADIAN = 57.295776
    xij = i[0] - j[0];
    yij = i[1] - j[1];
    zij = i[2] - j[2];

    xkj = k[0] - j[0];
    ykj = k[1] - j[1];
    zkj = k[2] - j[2];

    xkl = k[0] - l[0];
    ykl = k[1] - l[1];
    zkl = k[2] - l[2];

    dx = yij * zkj - zij * ykj;
    dy = zij * xkj - xij * zkj;
    dz = xij * ykj - yij * xkj;

    gx = ykj * zkl - zkj * ykl;
    gy = zkj * xkl - xkj * zkl;
    gz = xkj * ykl - ykj * xkl;

    bi = dx * dx + dy * dy + dz * dz;
    bk = gx * gx + gy * gy + gz * gz;
    ct = dx * gx + dy * gy + dz * gz;

    bi = math.sqrt(bi);
    bk = math.sqrt(bk);
    z1 = 1.0 / bi;
    z2 = 1.0 / bk;

#	// ct is the cosine of the angle between the normals to the two atom planes
    ct = ct * z1 * z2;
    if ct > 1.0:
        ct = 1.0;
    elif ct < -1.0:
        ct = -1.0;

#	// ap is the angle, in radians, between the two planes
    ap = math.acos(ct);
#	// the vector perpendicular to the normals to the two planes is compared
#	// with the direction of the central bond vector to determine the sign of
#	// the torsion

    si = xkj * (dy * gz - dz * gy) + ykj * (dz * gx - dx * gz) + zkj * (dx * gy - dy * gx);
    if si < 0.0:
        ap = -ap;

    return ap * RADIAN;

def get_rotatematrix(axis, origin, angle):
    angle = angle/180.0*3.1415926
    ccos = math.cos(angle)
    ssin = math.sin(angle)
                                                
    a = np.array(axis)
    u, v, w = a / np.sqrt(np.sum(np.dot(a,a)))
    a, b, c = origin
    rot_matrix = []
    #http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
    rot_matrix.append([u**2+(v**2+w**2)*ccos, u*v*(1-ccos)-w*ssin, u*w*(1-ccos)+v*ssin, \
            (a*(v**2+w**2)-u*(b*v+c*w))*(1-ccos)+(b*w-c*v)*ssin])
    rot_matrix.append([ u*v*(1-ccos)+w*ssin, v**2+(u**2+w**2)*ccos, v*w*(1-ccos)-u*ssin,\
        (b*(u**2+w**2)-v*(a*u+c*w))*(1-ccos)+(c*u-a*w)*ssin])
    rot_matrix.append([ u*w*(1-ccos)-v*ssin, v*w*(1-ccos)+u*ssin, w**2+(u**2+v**2)*ccos,\
            (c*(u**2+v**2)-w*(a*u+b*v))*(1-ccos)+(a*v-b*u)*ssin])
    rot_matrix.append([0, 0, 0, 1])
    rot_matrix = np.array(rot_matrix)
    return rot_matrix

def diss(cor1, cor2):
        dis = (cor1[0]-cor2[0])*(cor1[0]-cor2[0]);
        dis = dis+(cor1[1]-cor2[1])*(cor1[1]-cor2[1]);
        dis = dis+(cor1[2]-cor2[2])*(cor1[2]-cor2[2]);
        return math.sqrt(dis)

class topology:
        sidechain={}
        sidechain["ARG"]=[["N","CA","CB","CG"],["CA","CB","CG","CD"],["CB","CG","CD","NE"],["CG","CD","NE","CZ"]]
        sidechain["ASN"]=[["N","CA","CB","CG"],["CA","CB","CG","OD1"]]
        sidechain["ASP"]=[["N","CA","CB","CG"],["CA","CB","CG","OD1"]]
        sidechain["CYS"]=[["N","CA","CB","SG"]]
        sidechain["GLN"]=[["N","CA","CB","CG"],["CA","CB","CG","CD"],["CB","CG","CD","OE1"]]
        sidechain["GLU"]=[["N","CA","CB","CG"],["CA","CB","CG","CD"],["CB","CG","CD","OE1"]]
        sidechain["HIS"]=[["N","CA","CB","CG"],["CA","CB","CG","ND1"]]
        sidechain["ILE"]=[["N","CA","CB","CG1"],["CA","CB","CG1","CD1"]]
        sidechain["LEU"]=[["N","CA","CB","CG"],["CA","CB","CG","CD1"]]
        sidechain["LYS"]=[["N","CA","CB","CG"],["CA","CB","CG","CD"],["CB","CG","CD","CE"],["CG","CD","CE","NZ"]]
        sidechain["MET"]=[["N","CA","CB","CG"],["CA","CB","CG","SD"],["CB","CG","SD","CE"]]
        sidechain["PHE"]=[["N","CA","CB","CG"],["CA","CB","CG","CD1"]]
        sidechain["SER"]=[["N","CA","CB","OG"]]
        sidechain["THR"]=[["N","CA","CB","OG1"]]
        sidechain["TRP"]=[["N","CA","CB","CG"],["CA","CB","CG","CD1"]]
        sidechain["TYR"]=[["N","CA","CB","CG"],["CA","CB","CG","CD1"]]
        sidechain["VAL"]=[["N","CA","CB","CG1"]]
        sidechain["GLY"]=[]
        sidechain["ALA"]=[]
        sidechain["PRO"]=[]
        atomname = {}
        bbatom = ["N", "CA", "C", "O"]
        atomname["ARG"] = bbatom + ["CB","CG","CD","NE","CZ","NH1","NH2"]
        atomname["ASN"] = bbatom + ["CB","CG","OD1","ND2"]
        atomname["ASP"] = bbatom + ["CB","CG","OD1","OD2"]
        atomname["CYS"] = bbatom + ["CB","SG"]
        atomname["GLN"] = bbatom + ["CB","CG","CD","OE1","NE2"]
        atomname["GLU"] = bbatom + ["CB","CG","CD","OE1","OE2"]
        atomname["HIS"] = bbatom + ["CB","CG","ND1","CD2","CE1","NE2"]
        atomname["ILE"] = bbatom + ["CB","CG1","CG2","CD1"]
        atomname["LEU"] = bbatom + ["CB","CG","CD1","CD2"]
        atomname["LYS"] = bbatom + ["CB","CG","CD","CE","NZ"]
        atomname["MET"] = bbatom + ["CB","CG","SD","CE"]
        atomname["PHE"] = bbatom + ["CB","CG","CD1","CD2","CE1","CE2","CZ"]
        atomname["SER"] = bbatom + ["CB","OG"]
        atomname["THR"] = bbatom + ["CB","OG1","CG2"]
        atomname["TRP"] = bbatom + ["CB","CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"]
        atomname["TYR"] = bbatom + ["CB","CG","CD1","CD2","CE1","CE2","CZ","OH"]
        atomname["VAL"] = bbatom + ["CB","CG1","CG2"]
        atomname["GLY"] = bbatom 
        atomname["ALA"] = bbatom + ["CB"]
        atomname["PRO"] = bbatom + ["CG", "CD"]

def three2one(resname):
        aaname = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
        'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
        'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
        'TRP':'W','TYR':'Y'};
        try:
            aa = aaname[resname]
        except:
                #                    if self.restype == "RES":
                #print "unknow resname %s"%resname
            aa = "X"
        return aa


aaname_global = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
        'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
        'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
        'TRP':'W','TYR':'Y'};

def one2three(resname):
        aaname = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
        'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
        'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
        'TRP':'W','TYR':'Y'};
        bb = {}
        for i in aaname.keys():
            bb[aaname[i]] = i
        try:
            aa = bb[resname]
        except:
            aa = "XXX"
        return aa


class atom:
    mass = {"H":1.008, "C":12.011, "O":15.999, "N":14.007, \
        "P":30.974, "S":32.060, "CA":40.08, "F":18.998, "ZN":65.37,\
        "CL":35.45, "AU":196.9665,"CO":58.9332,"CS":132.9055,"CU":63.546,\
        "FE":55.845, "K":39.0983, "MG":24.305, "MN":54.938, "NA":22.9897,\
        "PB":207.2,"W":183.84,"NI":58.693,"I":126.904,"HO":164.930,"IR":192.217,
	"AG":107.868,"AL":26.981,"AS":74.921,"B":10.81,"BA":137.327,"BE":9.012,\
	"BR":79.904,"CD":112.414,"CR":51.996,"D":1.008,"GA":69.723,"GD":157.25,\
	"HG":200.592,"HO":164.930,"I":126.904,"IR":192.217,"LI":6.94,"LU":174.9668,\
	"MO":95.95,"NI":58.6934,"PA":231.03588,"PR":140.90766,"PT":195.084,"RU":101.07\
	,"SE":78.971,"SM":150.36,"SR":87.62,"TB":158.925,"TE":127.60,"TL":204.38,\
	"U":238.0289,"V":50.9415,"XE":131.293,"Y":88.90584,"YB":173.045}

    def __init__(self, atomtype, atomid, atomname_orig, resname, chainid, resid,\
            x=999, y=999, z=999, occu=1, bfactor=0, element="X"):
        self.atomtype = atomtype
        self.atomid = atomid
        self.atomname_orig = atomname_orig
        self.atomname = atomname_orig.strip(" ")
        self.resname = resname
        self.chainid = chainid
        self.resid = resid
        self.coor = np.array([x, y, z, 1], dtype=np.float)
        self.occu = occu
        self.bfactor = bfactor
        self.element = element.upper()
        if not self.element in atom.mass:
            self.mass = -1
            #print "atom mass type not found %s"%self.element
            #raise Exception("atom mass type not found %s"%self.element)
        else:
            self.mass = atom.mass[self.element]

    def set_coor(self, x, y, z):
        self.coor[0] = x
        self.coor[1] = y
        self.coor[2] = z

    def get_coor(self):
        return np.array(self.coor[0:3])
    
    def get_chainid(self):
        return self.chainid

    def get_atomid(self):
        return self.atomid

    def get_resid(self):
        return self.resid
    
    def get_atomname(self):
        return self.atomname

    def get_atomname_orig(self):
        return self.atomname_orig
    
    def get_resname(self):
        return self.resname
    
    def get_bfactor(self):
        return self.bfactor
    
    def get_occu(self):
        return self.occu

    def rotate_by(self, rot_matrix):
        self.coor = np.dot(self.coor, rot_matrix.T)
    
    def backup_coor(self):
        self.coor_backup = np.copy(self.coor)
    
    def recover_coor(self):
        self.coor = np.copy(self.coor_backup)

    def move_by(self, coor):
        self.coor[0] = self.coor[0] + coor[0]
        self.coor[1] = self.coor[1] + coor[1]
        self.coor[2] = self.coor[2] + coor[2]

class residue:
        aaname = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
        'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
        'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
        'TRP':'W','TYR':'Y'}
        dnarna = ["DA", "DU", "DC", "DG", "DT"]
 
        def __init__(self, resname, resid, chainid):
            self.resname = resname
            self.resid = resid
            self.chainid = chainid
            if not self.resname in topology.atomname:
                self.pdbatoms = []
            else:
                self.pdbatoms = topology.atomname[self.resname][:]
            self.atom = []
            self.atomnum = 0
            self.atomnamelist = []
            self.resname_key = self.make_reskey(self.resname, self.resid, self.chainid)

            if self.resname in residue.aaname:
                self.moltype = "prot"
            elif self.resname in residue.dnarna:
                self.moltype = "dnarna"
            elif self.resname == "HOH":
                self.moltype = "water"
            else:
                self.moltype = "other"

            self.minoccu = 1
            self.minbboccu = 1
            self.isorient = False
            self.sasa = -1
            self.dssp = "none"
        
        def make_reskey(self, resname, resid, chainid):
            resname = resname.replace(" ","")
            resid = resid.replace(" ","")
            chainid = chainid.replace(" ","")
            return "_".join([resname, resid, chainid])

        def get_resname(self):
            return self.resname
        def get_resname1(self):
            return three2one(self.get_resname)

        def set_sasa(self, sasa):
            self.sasa = sasa

        def get_sasa(self):
            return self.sasa

        def add_atom(self, atm):
            self.atom.append(atm)
            self.atomnum += 1
            assert self.resname == atm.get_resname()
            assert self.chainid == atm.get_chainid()
            assert self.resid == atm.get_resid()
            atomname = atm.get_atomname()
            self.atomnamelist.append(atomname)
            if atomname in self.pdbatoms:
                self.pdbatoms.remove(atomname)

            occu = atm.get_occu()
            if occu < self.minoccu:
                self.minoccu = occu
            if atomname in ["N", "C", "CA"]:
                if occu < self.minbboccu:
                    self.minbboccu = occu

        def orient(self):
            #move Ca to x,y,z = (0,0,0)
            #align N to -x axis, align C to x-y plane and make CB at z > 0
            if self.isorient:
                self.move_by(self.orient_displace)
                self.apply_rotmatrix(self.orient_rotatematrix)
                return self.orient_displace, self.orient_rotatematrix

            cacoor = self.get_coor("CA")
            self.move_by(-cacoor)
            ncoor = self.get_coor("N")
            if ncoor[0] != 0 and ncoor[1] == 0 and ncoor[1] ==0:
                ncoor[1] = 1e-6
                ncoor[2] = 1e-6
            ncoor_len = np.sqrt(np.sum(np.dot(ncoor, ncoor)))
            ncoor_norm = ncoor / ncoor_len
            xaxis = np.array([-1, 0, 0])
            cross = np.cross(ncoor_norm, xaxis)
            cos = np.dot(ncoor_norm, xaxis)
            if math.fabs(cos-1) < 1e-5:
                cos = 1
            if math.fabs(cos-(-1)) < 1e-5:
                cos = -1
            angle_nx = np.arccos(cos) * 180 / np.pi
            if math.isnan(angle_nx):
                raise Exception("value NaN in arccos %f"%cos)

            cross1 = np.cross(xaxis, cross)
            cos = np.dot(ncoor_norm, cross1)
            if math.fabs(cos-1) < 1e-5:
                cos = 1
            if math.fabs(cos-(-1)) < 1e-5:
                cos = -1
            angle1 = np.arccos(cos) * 180 / np.pi
            if math.isnan(angle1):
                raise Exception("value NaN in arccos %f"%cos)

            if angle1 < 90:
                rotangle = angle_nx
            else:
                rotangle = 360 - angle_nx
               
            rot_matrix = get_rotatematrix(cross, [0,0,0], rotangle)
            self.apply_rotmatrix([rot_matrix])


            #now put C in the x-y plane
            ccoor = self.get_coor("C")
            cos = ccoor[1] / np.sqrt(ccoor[1]**2 + ccoor[2]**2)
            if math.fabs(cos-1) < 1e-5:
                cos = 1
            if math.fabs(cos-(-1)) < 1e-5:
                cos = -1
            angle = np.arccos(cos) * 180 / np.pi
            if math.isnan(angle):
                raise Exception("value NaN in arccos %f"%cos)
            if ccoor[2] > 0 :
                angle = 180 - angle
            if ccoor[2] < 0:
                angle = 180 + angle
            rot_matrix1 = get_rotatematrix([1,0,0], [0,0,0], angle)
            self.apply_rotmatrix([rot_matrix1])

            self.isorient = True
            self.orient_displace = np.copy(-cacoor)
            self.orient_rotatematrix = [rot_matrix, rot_matrix1]

            return self.orient_displace, self.orient_rotatematrix 

        def backup_coor(self):
            for atm in self.atom:
                atm.backup_coor()

        def recover_coor(self):
            for atm in self.atom:
                atm.recover_coor()

        def apply_rotmatrix(self, rot_matrix):
            for atm in self.atom:
                for rot in rot_matrix:
                    atm.rotate_by(rot)

        def move_by(self, coor):
            for atm in self.atom:
                atm.move_by(coor)

        def has_occuless1(self, bbonly = False):
            if bbonly:
                if self.minbboccu < 1:
                    return True
                else:
                    return False
            else:
                if self.minoccu < 1:
                    return True
                else:
                    return False
        def has_atom(self, atomlist):
            for a in atomlist:
                if not a in self.atomnamelist:
                    return False
            return True

        def has_missingatom(self, bbonly = False):
            if bbonly:
                if len(self.pdbatoms) > 0 and ("N" in self.pdbatoms and "C" in self.pdbatoms and \
                        "CA" in self.pdbatoms):
                    #print self.pdbatoms, "missing in", self.resname_key
                    return True
                else:
                    return False
            else:
                if len(self.pdbatoms) > 0:
                    return True
                else:
                    return False

        def get_coor(self, atomname):
            if not atomname in self.atomnamelist:
                raise Exception("atom name %s does not exist in residue %s"%(atomname, self.resname_key))
            i = self.atomnamelist.index(atomname)
            return self.atom[i].get_coor()

        def set_neib_res(self, res, tag="prev"):
            if tag == "prev":
                self.prev_res = res
            elif tag == "next":
                self.next_res = res
            else:
                raise Exception("unknown tag in set_neib_res")

        def get_chainid(self):
            return self.atom[0].get_chainid()

        def get_dihedral(self, typee):
            if typee == "phi":
                return self.phi
            elif typee == "psi":
                return self.psi
            elif typee == "omega":
                return self.omega
            else:
                raise Exception("unknown dihedral type %s"%typee)

        def get_resname_key(self):
            return self.resname_key

        def get_dssp(self):
            return self.dssp
        
        def set_dssp(self, dssp):
            self.dssp = dssp

        def get_chi(self):
            return self.chi

               
        def iscontact(self, res2, cutoff = 4.5):
                for at1 in self.atom:
                    cor1 = at1.get_coor()
                    for at2 in res2.atom:
                            cor2 = at2.get_coor()
                            dis = (cor1[0]-cor2[0])*(cor1[0]-cor2[0])
                            dis = dis+(cor1[1]-cor2[1])*(cor1[1]-cor2[1])
                            dis = dis+(cor1[2]-cor2[2])*(cor1[2]-cor2[2])
                            if dis > 35 * 35:
                                return [0, dis]
                            if dis < cutoff * cutoff:
                                return [1, dis]
                return [0, dis]

#COLUMNS        DATA  TYPE    FIELD        DEFINITION
#-------------------------------------------------------------------------------------
#   1 -  6        Record name   "ATOM  "
#   7 - 11        Integer       serial       Atom  serial number.
#  13 - 16        Atom          name         Atom name.
#  17             Character     altLoc       Alternate location indicator.
#  18 - 20        Residue name  resName      Residue name.
#  22             Character     chainID      Chain identifier.
#  23 - 26        Integer       resSeq       Residue sequence number.
#  27             AChar         iCode        Code for insertion of residues.
#  31 - 38        Real(8.3)     x            Orthogonal coordinates for X in Angstroms.
#  39 - 46        Real(8.3)     y            Orthogonal coordinates for Y in Angstroms.
#  47 - 54        Real(8.3)     z            Orthogonal coordinates for Z in Angstroms.
#  55 - 60        Real(6.2)     occupancy    Occupancy.
#  61 - 66        Real(6.2)     tempFactor   Temperature  factor.
#  77 - 78        LString(2)    element      Element symbol, right-justified.
#  79 - 80        LString(2)    charge       Charge  on the atom.
                
        def writePDB(self, fp):
            for i in range(0,self.atomnum):
                coor = self.atom[i].get_coor()
                fp.write("ATOM  %5d %-4s %-3s %c%5s   %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(self.atom[i].get_atomid(),\
                        self.atom[i].get_atomname_orig(),self.resname,self.chainid,self.resid,\
                        coor[0],coor[1],coor[2],self.atom[i].get_occu(),self.atom[i].get_bfactor()))

class protein:
        def __init__(self, pdbfile="", resname4 = False):
                #resname4: resname is 4 letter, used for lipids, DHPC
                self.atomlist = []
                self.residuelist = []
                self.seqres = {}
                if pdbfile != "":
                    self.readPDB(pdbfile, resname4)
                self.pdbfile = pdbfile

        def readPDB(self,pdbfile ="", resname4=False):
                fin = open(pdbfile, 'r')
                resid_prev = "null"
                chain_prev = "null"
                residueAltKey = {}
                nmodel = 0
                for line in fin.readlines():
                        line = line.strip('\n')
                        linee = line.split()
                        if len(linee) == 0:
                            continue
                        if len(linee) >= 2 and linee[0] == "MODEL" and linee[1].isdigit():
                            nmodel += 1
                            if nmodel >= 2:
                                break
                        if linee[0] == "SEQRES" and len(linee) >= 3:
                            #SEQRES  10 A  245  THR LYS ASN ILE VAL TYR PRO PHE ASP GLN TYR ILE ALA          
                            chain = linee[2]
                            seq = linee[4:]
                            if not chain in self.seqres:
                                self.seqres[chain] = []
                            self.seqres[chain].extend(seq)

                        if len(line) >= 6 and (line[0:4] == "ATOM" or line[0:6] == "HETATM"):
                                ptype = line[0:4]
                                #if len(line) >=  78 and line[77] == "H":
                                #    continue
                                if resname4 == False:
                                    pdb_resname = line[17:20]
                                else:
                                    pdb_resname = line[17:21]
                                if pdb_resname.replace(" ","") == "HOH":
                                    continue
                                pdb_resid = line[22:27]
                                pdb_chain = line[21]
                                if pdb_chain == " ":
                                    pdb_chain = "A"
                                pdb_coorx = float(line[30:38])
                                pdb_coory = float(line[38:46])
                                pdb_coorz = float(line[46:54])
                                try:
                                    pdb_Bfactor = float(line[60:66])
                                except:
                                    pdb_Bfactor = 0
                                try:
                                    pdb_occu = float(line[54:60])
                                except:
                                    pdb_occu = 1.0
                                pdb_atomname_orig = line[12:16]
                                pdb_atomname = pdb_atomname_orig.strip(" ")
                                if pdb_atomname == "OT1":
                                    pdb_atomname = "O"
                                    pdb_atomname_orig = "O"
                                if pdb_atomname[0] == "H":
                                    continue
                                if pdb_resname == "ILE" and pdb_atomname == "CD":
                                    pdb_atomname = "CD1"
                                    pdb_atomname_orig = " CD1"
                                pdb_atomid = int(line[6:11])
                                #pdb_element = line[76:78].replace(" ","")
                                pdb_element = pdb_atomname[0]
                                if pdb_resname == "MSE" and pdb_atomname == "SE":
                                    pdb_atomname = "SD"
                                    pdb_atomname_orig = " SD "
                                    pdb_element = "S"
                                if pdb_resname == "MSE":
                                    ptype = "ATOM"
                                    pdb_resname = "MET"

                                alt = line[16]
                                if not (alt == " " or alt == "A"):
                                    continue
                                if ptype == "HETA":
                                    continue
                                if not pdb_resname.replace(" ", "") in aaname_global:
                                    continue
                    
                                atm = atom(linee[0], pdb_atomid, pdb_atomname_orig, pdb_resname, \
                                        pdb_chain, pdb_resid, pdb_coorx, pdb_coory, pdb_coorz, pdb_occu, pdb_Bfactor, pdb_element)
                                self.atomlist.append(atm)
                fin.close()
                self.natom = len(self.atomlist)
                self.build_residue()
                self.build_chain()

        def build_chain(self):
            if len(self.seqres.keys()) == 0:
                for i in range(self.residuenum):
                    ch = self.residuelist[i].get_chainid()
                    if not ch in self.seqres:
                        self.seqres[ch] = []
                    self.seqres[ch].append(self.residuelist[i].get_resname1())
            #group uniq chains into groups
            self.chain_groups = []
            for chainid in self.seqres:
                seq = self.seqres[chainid]
                tag = False
                for i in range(0, len(self.chain_groups)):
                    for ch in self.chain_groups[i]:
                        if seq == self.seqres[ch]:
                            self.chain_groups[i].append(chainid)
                            tag = True
                            break
                if tag == False:
                    self.chain_groups.append([chainid])
            
            #find uniq chain-chain pairs
            self.uniq_chainpair = []
            for i in range(0, len(self.chain_groups)):
                self.uniq_chainpair.append(self.chain_groups[i][0] + self.chain_groups[i][0])
                for j in range(1, len(self.chain_groups[i])):
                    self.uniq_chainpair.append(self.chain_groups[i][0] + self.chain_groups[i][j])

                for j in range( i+1, len(self.chain_groups)):
                    for ch in self.chain_groups[j]:
                        self.uniq_chainpair.append(self.chain_groups[i][0] + ch)
            #print "chain groups", self.chain_groups
            #print "uniq chain pairs", self.uniq_chainpair
        
        def build_residue(self):
            atoms = []
            chainid_prev = "xxx"
            resid_prev = "xxx"
            residueatomid = []
            for i in range(0, self.natom):
                chainid = self.atomlist[i].get_chainid()
                resid = self.atomlist[i].get_resid()
                if chainid != chainid_prev:
                    if chainid_prev != "xxx":
                        #res = residue(atoms)
                        residueatomid.append(atoms[:])
                        atoms = []
                else:
                    if resid != resid_prev:
                        #res = residue(atoms)
                        #self.residuelist.append(res)
                        residueatomid.append(atoms[:])
                        atoms = []
                atoms.append(i)
                chainid_prev = chainid
                resid_prev = resid
            #res = residue(atoms)
            #self.residuelist.append(res)
            residueatomid.append(atoms[:])
            for i in residueatomid:
                resname = self.atomlist[i[0]].get_resname()
                resid = self.atomlist[i[0]].get_resid()
                chainid = self.atomlist[i[0]].get_chainid()
                res = residue(resname, resid, chainid)
                for j in i:
                    res.add_atom(self.atomlist[j])
                self.residuelist.append(res)

            self.residuemap = {}
            for i in range(0, len(self.residuelist)):
                self.residuemap[self.residuelist[i].get_resname_key()] = i
                if i == 0:
                    self.residuelist[i].set_neib_res(None, "prev")
                else:
                    if self.residuelist[i-1].get_chainid() != self.residuelist[i].get_chainid():
                        self.residuelist[i].set_neib_res(None, "prev")
                    else:
                        self.residuelist[i].set_neib_res(self.residuelist[i-1], "prev")

                if i == len(self.residuelist) - 1:
                    self.residuelist[i].set_neib_res(None, "next")
                else:
                    if self.residuelist[i+1].get_chainid() != self.residuelist[i].get_chainid():
                        self.residuelist[i].set_neib_res(None, "next")
                    else:
                        self.residuelist[i].set_neib_res(self.residuelist[i+1], "next")
            self.residuenum = len(self.residuelist)


        def get_residueindex(self, resname, resid, chainid):
            resname_key = self.residuelist[0].make_reskey(resname, resid, chainid)
            if not resname_key in self.residuemap:
                raise Exception("residue %s does not exist"%resname_key)
            else:
                return self.residuemap[resname_key]

        def writePDB(self, filename, list=[]):
                fp = open(filename, 'w');
                if len(list)==0:
                        list = range(0,self.residuenum)
                for i in list:
                    if self.residuelist[i].has_atom(["CA", "C", "N", "O"]) == True:
                        self.residuelist[i].writePDB(fp)
                    else:
                        print (self.pdbfile, "residue %s has missing backbone atoms, skip"%self.residuelist[i].resname_key)
                fp.close();

        def get_fasta(self):
                seq=""
                for i in self.residuelist:
                        seq = seq + i.resname1
                return seq

if __name__ == "__main__":
    pdbfile = sys.argv[1]
    output = sys.argv[2]
    prot = protein(pdbfile)
    prot.writePDB(output)


