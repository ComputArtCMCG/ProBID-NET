#!/export/apps/bin/python
import sys
import math
import string
import numpy as np
import os
import h5py
from Bio import pairwise2 
import copy
import itertools
CUTOFF = 25

def identity(s1, s2):
    a = pairwise2.align.globalxx(s1, s2, one_alignment_only=True) 
    count = 0
    for i in range(len(a[0][0])):
        if a[0][0][i] != "-" and a[0][1][i] != "-" and a[0][0][i] == a[0][1][i]:
            count += 1
    if len(a[0][0]) < len(a[0][1]):
        return float(count)/len(a[0][0])
    else:
        return float(count)/len(a[0][1])

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

def angle(i, j, k):
    i = np.array(i)
    j = np.array(j)
    k = np.array(k)
    RADIAN = 57.295776
    dij = i - j
    dkj = k - j
    d1 = np.sqrt(np.sum(dij*dij))
    d2 = np.sqrt(np.sum(dkj*dkj))
    dotcos = np.sum(dij*dkj)/(d1*d2)
    return math.acos(dotcos) * RADIAN

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
        print ("resname %s not recognized, convert to XXX"%resname)
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

    def __init__(self, atomtype, atomid, atomname, resname, chainid, resid,\
            x=999, y=999, z=999, occu=1, bfactor=0, element="X"):
        self.atomtype = atomtype
        self.atomid = atomid
        self.atomname = atomname
        self.resname = resname#3
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
    
    def get_resname(self):
        return self.resname
    
    def get_bfactor(self):
        return self.bfactor
    
    def get_occu(self):
        return self.occu

    def rotate_by(self, rot_matrix): #0
        self.coor = np.dot(self.coor, rot_matrix.T)
    
    def backup_coor(self):
        self.coor_backup = np.copy(self.coor)
    
    def recover_coor(self):
        self.coor = np.copy(self.coor_backup)

    def move_by(self, coor):
        self.coor[0] = self.coor[0] + coor[0]
        self.coor[1] = self.coor[1] + coor[1]
        self.coor[2] = self.coor[2] + coor[2]

    def set_resname(self,resname):
        self.resname=resname

    def printPDB(self):
        coor = self.get_coor()
        print ("ATOM  %5s %-4s %-3s %c%5s   %8.3f%8.3f%8.3f%6.2f%6.2f"%(self.get_atomid(),\
                self.get_atomname(),self.get_resname(),self.get_chainid(),self.get_resid(),\
                coor[0],coor[1],coor[2],self.get_occu(),self.get_bfactor()  ))
    def writePDB(self, fp):
        coor = self.get_coor()
        fp.write("ATOM  %5s %-4s %-3s %c%5s   %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(self.get_atomid(),\
                self.get_atomname(),self.get_resname(),self.get_chainid(),self.get_resid(),\
                coor[0],coor[1],coor[2],self.get_occu(),self.get_bfactor()  ))

class residue:
        aaname = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
        'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
        'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
        'TRP':'W','TYR':'Y'}
        dnarna = ["DA", "DU", "DC", "DG", "DT"]
 
        def __init__(self, resname, resid, chainid):
            self.resname = resname#3
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
        def get_resid(self):
            return self.resid
        def get_resname1(self):
            return three2one(self.get_resname)

        def set_sasa(self, sasa):
            self.sasa = sasa

        def get_sasa(self):
            return self.sasa

        def add_atom(self, atm):  #1
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

        def orientCBonZ(self):
            #move Ca to x,y,z = (0,0,0)
            #align CB to +Z axis, move N to y=0 plane
            if self.isorient:
                self.move_by(self.orient_displace)
                self.apply_rotmatrix(self.orient_rotatematrix)
                return self.orient_displace, self.orient_rotatematrix

            cacoor = self.get_coor("CA")
            self.move_by(-cacoor)
            
            cbcoor = self.get_coor("CB")
            zaxis = np.array([0,0,1])
            origin = np.zeros(3)
            initangle = angle(cbcoor, origin, zaxis)
            anglediff = initangle
            rotmatrix = get_rotatematrix(np.cross(cbcoor, zaxis), origin, anglediff)
            self.apply_rotmatrix([rotmatrix])
            cbcoor = self.get_coor("CB")
            assert np.fabs(cbcoor[0]) < 1e-4
            assert np.fabs(cbcoor[1]) < 1e-4
            ncoor = self.get_coor("N")
            ncoorOnXY = np.array([ncoor[0], ncoor[1], 0])
            xaxis_minus = np.array([-1,0,0])
            initangle = angle(ncoorOnXY, origin, xaxis_minus)
            anglediff = initangle
            if ncoorOnXY[1] < 0:
                anglediff = 360-anglediff
            rotmatrix1 = get_rotatematrix(zaxis, origin, anglediff)
            self.apply_rotmatrix([rotmatrix1])
            ncoor = self.get_coor("N")
            assert np.fabs(ncoor[1]) < 1e-4

            return -cacoor, [rotmatrix, rotmatrix1]

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
            if ncoor[0] != 0 and ncoor[1] == 0 and ncoor[2] ==0:
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

        def has_missingatom(self, bbonly = False):
            if bbonly:
                if len(self.pdbatoms) > 0 and ("N" in self.pdbatoms or "C" in self.pdbatoms or \
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
        
        def build_CB(self):
            for i in range(len(self.atom)):
                if self.atom[i].get_atomname().replace(" ", "") == "CB":
                    #print ("CB atom exists in residue %s, skip"%(self.get_resname_key()))
                    return 0
            cacoor = self.get_coor("CA")
            cbatm = copy.deepcopy(self.atom[0])
            cbatm.atomname = "CB"
            self.add_atom(cbatm)
            cbindex = len(self.atom) - 1 
            self.atom[cbindex].set_coor(cacoor[0], cacoor[1], cacoor[2])
            self.atom[cbindex].move_by([1.55, 0, 0])
            cbcoor = self.get_coor("CB")
            ccoor = self.get_coor("C")
            axisnorm = np.cross(ccoor-cacoor, cbcoor-cacoor)
            if np.max(np.fabs(axisnorm)) < 1e-6: #c, ca and cb on a line
                for i in range(len(self.atom)):
                    if self.atom[i].get_atomname().replace(" ", "") == "C":
                        self.atom[i].move_by([0.001, 0.001, 0.001])
                        ccoor = self.get_coor("C")
            initangle = angle(ccoor, cacoor, cbcoor)
            anglediff = 110.5 - initangle
            axisnorm = np.cross(ccoor-cacoor, cbcoor-cacoor)
            rotmatrix = get_rotatematrix(axisnorm, cacoor, anglediff)
            self.atom[cbindex].rotate_by(rotmatrix)

            ncoor = self.get_coor("N")
            cbcoor = self.get_coor("CB")
            initdihe = torsion_angle(ncoor, ccoor, cacoor, cbcoor)
            anglediff = 122.55 - initdihe
            rotmatrix = get_rotatematrix(cacoor-ccoor, ccoor, anglediff)
            self.atom[cbindex].rotate_by(rotmatrix)
            cbcoor = self.get_coor("CB")
            assert np.fabs(angle(ccoor, cacoor, cbcoor) - 110.5) < 0.01
            assert np.fabs(torsion_angle(ncoor, ccoor, cacoor, cbcoor) - 122.55) < 0.01
            assert np.fabs(diss(cacoor, cbcoor) - 1.55 ) < 0.01

        def set_neib_res(self, res, tag="prev"):
            if tag == "prev":
                self.prev_res = res
            elif tag == "next":
                self.next_res = res
            else:
                raise Exception("unknown tag in set_neib_res")

        def get_chainid(self):
            return self.atom[0].get_chainid()

        def calc_phipsi(self):
                #phi C(i-1)-N-CA-C
                #psi N-CA-C-N(i+1)
            if self.prev_res == None:
                self.phi = 999
            elif self.moltype != "prot" or self.prev_res.moltype != "prot":
                self.phi = 999
            else:
                try:
                    ci1coor = self.prev_res.get_coor("C")
                    ncoor = self.get_coor("N")
                    cacoor = self.get_coor("CA")
                    ccoor = self.get_coor("C")
                    dis = diss(ci1coor, ncoor)
                    if dis > 2:
                        self.phi = 999
                    else:
                    	self.phi = torsion_angle(ci1coor, ncoor, cacoor, ccoor)
                except:
                    self.phi = 999

            if self.next_res == None:
                self.psi = 999
            elif self.moltype != "prot" or self.next_res.moltype != "prot":
                self.psi = 999
            else:
                try:
                    ncoor = self.get_coor("N")
                    cacoor = self.get_coor("CA")
                    ccoor = self.get_coor("C")
                    n1coor = self.next_res.get_coor("N")
                    dis = diss(ccoor, n1coor)
                    if dis > 2:
                        self.psi = 999
                    else:
                        self.psi = torsion_angle(ncoor, cacoor, ccoor, n1coor)
                except:
                    self.psi = 999

        def get_dihedral(self, typee):
            if typee == "phi":
                return self.phi
            elif typee == "psi":
                return self.psi
            elif typee == "omega":
                return self.omega
            else:
                raise Exception("unknown dihedral type %s"%typee)

        def calc_omega(self): #0
            if self.prev_res == None:
                self.omega = 999
            elif self.moltype != "prot" or self.prev_res.moltype != "prot":
                self.omega = 999
            else:
                try:
                    ci1coor = self.prev_res.get_coor("CA")
                    c1coor = self.prev_res.get_coor("C")
                    ncoor = self.get_coor("N")
                    cacoor = self.get_coor("CA")
                    dis = diss(c1coor, ncoor)
                    if dis > 2:
                        self.omega = 999
                    else:
                        self.omega = torsion_angle(ci1coor, c1coor, ncoor, cacoor)
                except:
                    self.omega = 999

        def get_resname_key(self): 
            return self.resname_key

        def get_dssp(self):
            return self.dssp
        
        def set_dssp(self, dssp):
            self.dssp = dssp

        def get_chi(self):
            return self.chi

        def calc_chi(self):
                self.chi = []
                if not self.resname in topology.sidechain:
                    sdatom = []
                else:
                    sdatom = topology.sidechain[self.resname]
                self.nchi = len(sdatom)
                if self.nchi == 0:
                    self.chi = [999]
                for i in range(0, self.nchi):
                    try:
                        c1 = self.get_coor(sdatom[i][0])
                        c2 = self.get_coor(sdatom[i][1])
                        c3 = self.get_coor(sdatom[i][2])
                        c4 = self.get_coor(sdatom[i][3])
                        self.chi.append(torsion_angle(c1, c2, c3, c4))
                    except:
                        print ("chi %s does not exist in %s"%("-".join(sdatom[i]), self.resname_key))
                        self.chi.append(999)
               
        def iscontact(self, res2, cutoff = 4.5): #1
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

        def set_resname(self, resname):

            self.resname = resname
            for atm in self.atom:
                atm.set_resname(resname)





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
                
        def printPDB(self):
            for i in range(0,self.atomnum):
                coor = self.atom[i].get_coor()
                print ("ATOM  %5d %-4s %-3s %c%5s   %8.3f%8.3f%8.3f%6.2f%6.2f"%(self.atom[i].get_atomid(),\
                        self.atom[i].get_atomname(),self.resname,self.chainid,self.resid,\
                        coor[0],coor[1],coor[2],self.atom[i].get_occu(),self.atom[i].get_bfactor()))
        def writePDB(self, fp):
            for i in range(0,self.atomnum):
                '''if not self.atom[i].get_atomname() in ["N", "CA", "C", "CB", "O"]:
                    continue'''
                coor = self.atom[i].get_coor()
                fp.write("ATOM  %5d %-4s %-3s %c%5s   %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(self.atom[i].get_atomid(),\
                        self.atom[i].get_atomname(),self.resname,self.chainid,self.resid,\
                        coor[0],coor[1],coor[2],self.atom[i].get_occu(),self.atom[i].get_bfactor()))

class protein:
        def __init__(self, pdbfile="", resname4 = False):
                #resname4: resname is 4 letter, used for lipids, DHPC
                self.atomlist = []
                self.residuelist = []#1
                self.seqres = {}
                if pdbfile != "":
                    self.readPDB(pdbfile, resname4)
                self.pdbfile = pdbfile

        def readPDB(self,pdbfile ="", resname4=False):
                fin = open(pdbfile, 'r')
                resid_prev = "null"
                chain_prev = "null"
                residueAltKey = {}
                for line in fin.readlines():
                        line = line.strip('\n')
                        linee = line.split()
                        if len(linee) == 0:
                            continue
                        if linee[0] == "MODEL" and linee[1] == "2":
                            break
                        if linee[0] == "SEQRES":
                            continue
                            #SEQRES  10 A  245  THR LYS ASN ILE VAL TYR PRO PHE ASP GLN TYR ILE ALA          
                            chain = linee[2]
                            seq = linee[4:]
                            if not chain in self.seqres:
                                self.seqres[chain] = []
                            self.seqres[chain].extend(seq)

                        if line[0:4] == "ATOM" or line[0:6] == "HETATM":
                                if len(line) >=  78 and line[77] == "H":
                                    continue
                                if resname4 == False:
                                    pdb_resname = line[17:20]
                                else:
                                    pdb_resname = line[17:21]
                                if pdb_resname.replace(" ","") == "HOH":
                                    continue
                                if line[0:6] == "HETATM" and pdb_resname != "MSE":
                                    continue
                                pdb_resid = line[22:27]
                                pdb_chain = line[21]
                                pdb_coorx = float(line[30:38])
                                pdb_coory = float(line[38:46])
                                pdb_coorz = float(line[46:54])
                                try:
                                    pdb_Bfactor = float(line[60:66])
                                except:
                                    pdb_Bfactor = 0

                                pdb_occu = float(line[54:60])
                                pdb_atomname = line[11:16]
                                pdb_atomname = pdb_atomname.strip(" ")
                                if pdb_resname == "ILE" and pdb_atomname == "CD":
                                    pdb_atomname = "CD1"
                                pdb_atomid = int(line[6:11])
                                pdb_element = line[76:78].replace(" ","")
                                if pdb_resname == "MSE" and pdb_atomname == "SE":
                                    pdb_atomname = "SD"
                                    pdb_element = "S"
                                if pdb_resname == "MSE":
                                    pdb_resname = "MET"

                                #occu not 1
                                alt = line[16]
                                if not alt in ["A", " "]:
                                    continue
                                #tag = pdb_resid + "-" + pdb_chain
                                #if not tag in residueAltKey:
                                #    residueAltKey[tag] = alt
                                #if alt != residueAltKey[tag]:
                                #    continue
                    
                                atm = atom(linee[0], pdb_atomid, pdb_atomname, pdb_resname, \
                                        pdb_chain, pdb_resid, pdb_coorx, pdb_coory, pdb_coorz, pdb_occu, pdb_Bfactor, pdb_element)
                                self.atomlist.append(atm)
                fin.close()
                self.natom = len(self.atomlist)
                self.build_residue()
                self.build_chain()

        def build_chain(self):
            self.seqres1 = {}
            for r in self.residuelist:
                chain = r.chainid
                seq = r.resname
                if not chain in self.seqres:
                    self.seqres[chain] = []
                if not chain in self.seqres1:
                    self.seqres1[chain] = ""
                self.seqres[chain].append(seq)
                self.seqres1[chain] += three2one(seq)
            #group uniq chains into groups
            self.chain_groups = []
            for chainid in self.seqres:
                seq = self.seqres1[chainid]
                tag = False
                for i in range(0, len(self.chain_groups)):
                    for ch in self.chain_groups[i]:
                        if identity(seq, self.seqres1[ch]) > 0.95:
                            self.chain_groups[i].append(chainid)
                            tag = True
                            break
                if tag == False:
                    self.chain_groups.append([chainid])
 

        def build_chain_old(self):
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

                for j in range(i+1, len(self.chain_groups)):
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

        def load_sasa(self, sasafile):
            self.load_naccess(sasafile, 'ABS')

        def load_freesasa(self, sasfile):
            #load SAS results from freeSASA program
            for line in open(sasfile, "r").readlines():
                line = line.strip("\n")
                linee = line.split()
                if len(linee) > 0 and linee[0] == "SEQ":
                    chainid = linee[1]
                    resid = linee[2]
                    resname = linee[3]
                    sas = float(linee[5]) 
                    try:
                        index = self.get_residueindex(resname, resid, chainid)
                        self.residuelist[index].set_sasa(sas)
                    except:
                        raise Exception("error in load_sasa from file %s, no match for residue %s %s %s"%(sasfile, resname, resid, chainid))

            for i in range(0, len(self.residuelist)):
                if self.residuelist[i].get_sasa() < 0 and self.residuelist[i].moltype == "prot":
                    raise Exception("residue %s doe not have SASA value"%self.residuelist[i].get_resname_key())

        def get_residueindex(self, resname, resid, chainid):
            resname_key = self.residuelist[0].make_reskey(resname, resid, chainid)
            if not resname_key in self.residuemap:
                raise Exception("residue %s does not exist"%resname_key)
            else:
                return self.residuemap[resname_key]

        def printPDB(self, resid="null"):
                if resid=="null":
                    for i in self.residuelist:
                            i.printPDB()
                else:
                    self.residuelist[resid].printPDB()

        def writePDB(self, filename, list=[]):
                fp = open(filename, 'w');
                if len(list)==0:
                        list = range(0,self.residuenum)
                for i in list:
                    self.residuelist[i].writePDB(fp);
                fp.close();

        def get_fasta(self):
                seq=""
                for i in self.residuelist:
                        seq = seq + i.resname1
                return seq

        def load_dssp(self, dsspfile):
                fp = open(dsspfile, "r");
                nload = 0;
                for line in fp.readlines():
                    line = line.strip("\n")
                    if line[-1] == ".":
                            continue
                    if line.find("#") > 0:
                            continue;
                    if line[13] == "!":
                        continue
#  165  184AB G  S    S-     0   0    5    -23,-1.7     2,-0.4    -2,-0.4     7,-0.1  -0.983  83.4 -45.3 143.5-149.2   27.0   36.2   16.2

                    linee = line.split()
                    resid = line[5:11].replace(" ","")
                    chainid = line[11]
                    resname1 = line[13]
                    #The one letter code for the amino acid. If this letter is lower\
                    # case this means this is a cysteine that form a sulfur bridge with\
                    # the other amino acid in this column with the same lower case letter.
                    if resname1.islower():
                        resname1 = "C"
                    dsspcode = line[16]
                    if dsspcode == " ":
                            dsspcode = "space"
                    resname3 = one2three(resname1)
                    try:
                        index = self.get_residueindex(resname3, resid, chainid)
                        self.residuelist[index].set_dssp(dsspcode)
                    except:
                        if resname1 == "C":
                            try:
                                index = self.get_residueindex("CSS", resid, chainid)
                                self.residuelist[index].set_dssp(dsspcode)
                            except:
                                print ("%s %s %s %s in dssp file but not find in residuelist dssp"%(self.pdbfile, resname3, resid, chainid))
                        else:
                            #raise Exception("%s %s %s not find in setting dssp"%(resname3, resid, chainid) )
                            print ("%s %s %s %s in dssp file but not find in residuelist dssp"%(self.pdbfile, resname3, resid, chainid))

                for i in range(0, len(self.residuelist)):
                    if self.residuelist[i].get_dssp() == "none" and \
                            self.residuelist[i].moltype == "prot" and \
                            self.residuelist[i].has_missingatom() == False:
                        print ("#warning missing dssp info for %s"%self.residuelist[i].get_resname_key())
        def load_naccess(self, naccessfile, type = "REL"):
#REM                ABS   REL    ABS   REL    ABS   REL    ABS   REL    ABS   REL
#RES ILE A  22    84.27  48.1  22.86  16.6  61.40 165.2  25.14  18.1  59.12 164.3
#RES ASP A  23    79.52  56.6  71.04  69.2   8.47  22.5  42.51  86.3  37.00  40.6
#RES GLU A  24    97.06  56.3  91.16  67.7   5.90  15.7  54.76  90.8  42.30  37.8
#RES ASN A  25    49.23  34.2  44.20  41.6   5.03  13.3  27.65  59.8  21.58  22.1

                fp = open(naccessfile, "r")
                for line in fp.readlines():
                        line = line.strip("\n")
                        linee = line.split()
                        if linee[0] != "RES":
                                continue;
                        resname3 = linee[1]
                        chainid = line[8]
                        resid = line[9:14].replace(" ","")
                        if type == "REL":
                            sas = float(line[22:28].replace(" ",""))
                        else:
                            sas = float(line[15:22].replace(" ",""))
                            index = self.get_residueindex(resname3, resid, chainid)
                        try:
                            index = self.get_residueindex(resname3, resid, chainid)
                            self.residuelist[index].set_sasa(sas)
                        except:
                            print ("%s %s %s not find in setting Naccess"%(resname3, resid, chainid))

def isinbox(c1, c2, dis):
    a = np.fabs(c1-c2)
    if max(a) < dis:
        return True
    else:
        return False

def calc_gauss_density(d1, d2, r):
    dis = diss(d1, d2)
    return np.exp(-dis**2/(2*r**2))
    return np.exp(-dis**2/(2*r**2))/np.sqrt(2*np.pi)/r

def write_dx(data3d, boxsize, gridsize, output):
    assert len(boxsize) == 3
    origin = [-i/2.0 for i in boxsize]
    ngrid = [int(i/gridsize) for i in boxsize]
    fp = open(output, "w")
    fp.write("object 1 class gridpositions counts %d %d %d\n"%(ngrid[0], ngrid[1], ngrid[2]))
    fp.write("origin %f %f %f\n"%(origin[0], origin[1], origin[2]))
    fp.write("delta %f %f %f\ndelta %f %f %f\ndelta %f %f %f\n"%(gridsize,0,0, 0,gridsize,0, 0,0,gridsize))
    fp.write("object 2 class gridconnections counts %d %d %d\n"%(ngrid[0], ngrid[1], ngrid[2]))
    fp.write("object 3 class array type double rank 0 items %d data follows\n"%(ngrid[0]*ngrid[1]*ngrid[2]))
    count = 0
    for i in range(0, ngrid[0]):
        for j in range(0, ngrid[1]):
            for k in range(0, ngrid[2]):
                pot = data3d[i][j][k]
                fp.write("%f  "%pot)
                count = count + 1
                if count % 3 == 0:
                    fp.write("\n")
                    count = 0
    if count != 0:
        fp.write("\n")
    fp.write("object 4 class field\n")
    fp.close()
    print ("3D grid written to %s"%output)
    #fp.write("component \"positions\" value 1\n")
    #fp.write("component \"connections\" value 2\n")
    #fp.write("component \"data\" value 3\n")

def calc_CNNfeature(prot, reslist, boxsize, binsize, boxcenter, atomr_dict, atom_channel_dict):
    natomtype = len(set(list(atom_channel_dict.values())))
    grid = np.arange(-boxsize/2, boxsize/2+binsize, binsize)
    grid1 = np.arange(-boxsize/2-binsize, boxsize/2+2*binsize, binsize)
    gridcenter = [grid[i]*0.5+grid[i+1]*0.5 for i in range(len(grid)-1)]
    gridsize = len(gridcenter)
    neibgrid = list(itertools.product([-1, 0, 1], repeat=3))
    neibgrid = np.array(neibgrid)

    rescenter = prot.residuelist[reslist[0]].get_resname_key()
    data = np.zeros((gridsize, gridsize, gridsize, natomtype), dtype=np.float32)
    #print (prot.residuelist[reslist[0]].get_resname_key(), len(reslist))
    
    tag = 0
    for i in reslist:
        r = prot.residuelist[i]
        r.move_by(-boxcenter)
        for atm in r.atom:
            if tag == 0:
                resname = "CEN"
            else:
                resname = atm.resname
            atomname = atm.atomname
            key = (resname, atomname)
            if not key in atom_channel_dict:
                print (key, "not in atom_channel_dict, skip in calc CNN density")
                continue
            coor = atm.coor[0:3]
            if np.max(np.fabs(coor)) > boxsize/2.0 + 2*binsize:
                continue
            ind = np.array([np.searchsorted(grid1, c)-2 for c in coor])
            neibgridid = neibgrid + ind
            for neibid in neibgridid:
                if min(neibid) < 0 or max(neibid) > gridsize-1:
                    continue
                gridc = [gridcenter[neibid[0]], gridcenter[neibid[1]], gridcenter[neibid[2]]]
                density = calc_gauss_density(coor, gridc, atomr_dict[key])
                #print (coor, neibid, density, gridc) 
                data[neibid[0]][neibid[1]][neibid[2]][atom_channel_dict[key]-1] += density
        tag += 1

    return data

restypedict = {'ALA':1,'CYS':2,'ASP':3,'GLU':4,'PHE':5,'GLY':6,
        'HIS':7,'ILE':8,'LYS':9,'LEU':10,'MET':11,'ASN':12,
        'PRO':13,'GLN':14,'ARG':15,'SER':16,'THR':17,'VAL':18,
        'TRP':19,'TYR':20};
def get_restypeid(resname):
    if len(resname) == 1:
        resname = one2three(resname)
    id= restypedict[resname]-1
    return id

def get_cnndata(fpdb, target_reslist, boxsize, binsize, boxcenterZ, atomr_dict, atom_channel_dict):
    debug = False
    boxr = math.sqrt(3) * boxsize/2.0

    boxcenter = np.array([0,0,boxcenterZ])
    prot = protein(fpdb)
    dismatrix = np.zeros((len(prot.residuelist), len(prot.residuelist)), dtype=np.float)
    for i in range(0, len(prot.residuelist)):
        if prot.residuelist[i].get_resname() == "GLY":
            prot.residuelist[i].build_CB()
        prot.residuelist[i].backup_coor()
        try:
            cacoor1 = prot.residuelist[i].get_coor("CA")
        except:
            continue
        for j in range(i+1, len(prot.residuelist)):
            cacoor2 = prot.residuelist[j].get_coor("CA")
            dis = diss(cacoor1, cacoor2)
            dismatrix[i,j] = dis
            dismatrix[j,i] = dis
    dataall = []
    reskey = []
    resindex = [prot.get_residueindex(r[0], r[1], r[2]) for r in target_reslist] ####
    for i in resindex: ######################
        reslist = []
        displace, rot_matrix = prot.residuelist[i].orientCBonZ()
        cacoor = prot.residuelist[i].get_coor("CA")
        reslist.append(i)
        for j in range(0, len(prot.residuelist)):
            if j == i:
                continue
            if dismatrix[i][j] > boxr + 8:
                continue
            prot.residuelist[j].move_by(displace)
            prot.residuelist[j].apply_rotmatrix(rot_matrix)
            reslist.append(j)
        output = pdbfile+".%s"%prot.residuelist[i].get_resname_key()
        data = calc_CNNfeature(prot, reslist, boxsize, binsize, boxcenter, atomr_dict, atom_channel_dict)
        dataall.append(data)
        reskey.append(prot.residuelist[i].get_resname_key())
        print (prot.residuelist[i].get_resname_key(), data.shape)
        print ("atom density matrix", np.mean(data[:,:,:,0:-1]), np.max(data[:,:,:,0:-1]), np.min(data[:,:,:,0:-1]))
        if debug:
            prot.writePDB(output+".pdb", reslist)
            for j in range(0, data.shape[-1]-1):
                write_dx(data[:,:,:,j], [boxsize, boxsize, boxsize], binsize, 
                        pdbfile+".%s.ch%d.dx"%(prot.residuelist[i].get_resname_key(), j))
            write_dx(data[:,:,:,-1], [boxsize, boxsize, boxsize], binsize, 
                        pdbfile+".%s.apbs.dx"%(prot.residuelist[i].get_resname_key()))
            sys.exit()
        for j in reslist:
            prot.residuelist[j].recover_coor()
    return dataall, reskey

def get_reslist(f, ch):
    prot = protein(f)
    reslist = []
    for r in prot.residuelist:
        if r.get_chainid() != ch:
            continue
        if three2one(r.get_resname()) == "X":
            continue
        reslist.append((r.get_resname(), r.get_resid().replace(" ", ""), ch))
    return reslist

def get_atomr(fatomr):
    atomr_dict = {}
    atom_channel_dict = {}
    for line in open(fatomr):
        line = line.split()
        key = (line[0], line[1])
        rmin = float(line[3])
        index = int(line[4])
        atomr_dict[key] = np.sqrt(-rmin**2/np.log(0.05)/2)
        atom_channel_dict[key] = index
    return atomr_dict, atom_channel_dict


def change_chain(filename, chainid):
    prot = protein(filename)
    for r in prot.residuelist:
        m=r.get_chainid()
        if m == chainid:
            r.build_CB()
            r.set_resname('XXX')
    prot.writePDB(filename + '.'+chainid)
    print(filename+chainid)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print (sys.argv[0], "pdbidCH")
        sys.exit()
    filename=sys.argv[1]
    prot=protein(filename)
    chainID=[]
    for r in prot.residuelist:
        m=r.get_chainid()
        chainID.append(m)
    n=set(chainID)

    print(n)
    for i in n:
        change_chain(filename,i)



    '''f = h5py.File(pdbidCH + ".hdf5", "w")
    for i in zip(reskeyall, dataall):
        f.create_dataset(i[0], data=i[1], compression="lzf")
    f.close()
    print ("saved to %s"%(pdbid+".hdf5"))'''
