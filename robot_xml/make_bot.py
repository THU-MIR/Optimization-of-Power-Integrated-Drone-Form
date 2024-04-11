

from dm_control import mjcf
from dm_control import mujoco
import random

from datetime import datetime
import numpy as np
from lxml import etree
class robot_transform(object):
    def __init__(self,robot_file="lun_robot.xml",is_muscle=False) -> None:
        self.name=robot_file
        self.mjcf_model = mjcf.from_path("./robot_xml/"+robot_file)
        self.torso=self.mjcf_model.worldbody.body['torso']
        self.muscle_site={}
        self.direction=["lf","rf","lb","rb"]
        #self._ergodic_muscle_site()
        self.lf={}
        self.lb={}
        self.rf={}
        self.rb={}
        self.is_muscle=is_muscle
        for direction in self.direction:
            self._ergodic_leg(direction)
            if self.is_muscle :
                self._ergodic_muscle_site(direction)

    def reset(self):
        self.mjcf_model = mjcf.from_path("./robot_xml/"+self.name)

    def _ergodic_muscle_site(self,direction):
        for mn in range(1,4):
            for mn_n in range(1,5):
                site_name=direction+str(mn)+"_"+str(mn_n)
                leg=direction+"_leg"+str(mn)
                if mn ==1:
                    self.muscle_site[site_name]=self.torso.site[site_name]
                elif mn !=1:
                    self.muscle_site[site_name]=self.__dict__[direction][direction+"_leg"+str(mn-1)].site[site_name]
                site_name+="e"
                self.muscle_site[site_name]=self.__dict__[direction][leg].site[site_name]
    
    def _ergodic_leg(self,direction:str):
        leg1=direction+"_leg1"
        leg2=direction+"_leg2"
        leg3=direction+"_leg3"
        motor=direction+"_motor"
        self.__dict__[direction][leg1]=self.torso.body[leg1]
        self.__dict__[direction][leg2]=self.torso.body[leg1].body[leg2]
        self.__dict__[direction][leg3]=self.torso.body[leg1].body[leg2].body[leg3]
        self.__dict__[direction][motor]=self.torso.body[leg1].body[leg2].body[leg3].body[motor]
    
    def transform(self,p_args:list,bil_sym=False,cen_sym=False):
        #args身体变换参数
        #000：左前1腿长                 lf长020/010/000/####/100/110/120rf     
        #001：左前1腿宽                   宽021/011/001/####/101/111/121                               
        #010：左前2腿长                                ####
        #101：右前1腿宽                                ####
        #                                            ####
        #                             lb长220/210/200/####/300/310/320rb
        #                               宽221/211/201/####/301/311/321
        n=0
        args=np.zeros((4,3,1))
        for i in range(4):
            for g in range(3):
                    args[i][g][0]=np.clip(p_args[n],-0.9,1.0)/2
                    n+=1
                    
           
        if cen_sym:#中心对称
            args[1]=args[0]
            args[2]=args[0]
            args[3]=args[0]
        elif bil_sym:#镜面对称
            for i in range(3):
                args[1]=args[0]
                args[3]=args[2]

        self.__init__()

        for leg_n in range(4):
            self.leg_transform(leg_n,args[leg_n])
            #self.muscle_transform(leg_n,args[leg_n])
        return #mjcf.Physics.from_mjcf_model(self.mjcf_model)

    def leg_transform(self,leg_n:int,args):
        leg_half_long=0.5
        leg__half_wide=0.2
        leg=["lf","rf","lb","rb"]
        direction=leg[leg_n]
        leg_dict={}
        leg_dict[direction+"_motor_pos"]=self.__dict__[direction][direction+"_motor"].pos
        for i in range(1,4):
            name=direction+"_leg"+str(i)
            leg_dict[name+"_pos"]=self.__dict__[direction][name].pos
            leg_dict[name+"_fromto"]=self.__dict__[direction][name].geom[name+"_geom"].fromto
            leg_dict[name+"_joint_x"]=self.__dict__[direction][name].joint[name+"_Joint_x"].pos
            leg_dict[name+"_joint_z"]=self.__dict__[direction][name].joint[name+"_Joint_z"].pos
            leg_dict[name+"_size"]=self.__dict__[direction][name].geom[name+"_geom"].size
            leg_dict[name+"_fromto"][4]-leg_dict[name+"_fromto"][1]
            leg_dict[name+"_fromto"][1]-=args[i-1][0]
            leg_dict[name+"_fromto"][4]+=args[i-1][0]
            
            self.__dict__[direction][name].geom[name+"_geom"].fromto=leg_dict[name+"_fromto"]
            
        if leg_n ==0 or leg_n ==2:
            leg_dict[direction+"_leg1"+"_pos"][1]+=args[0][0]
            leg_dict[direction+"_leg2"+"_pos"][1]=leg_dict[direction+"_leg2"+"_pos"][1]+args[0][0]+args[1][0]
            leg_dict[direction+"_leg3"+"_pos"][1]=leg_dict[direction+"_leg3"+"_pos"][1]+args[1][0]+args[2][0]
            leg_dict[direction+"_motor_pos"][1]+=args[2][0]
            for i in range(3):
                leg_dict[direction+"_leg"+str(i+1)+"_joint_x"][1]-=args[i][0]
                leg_dict[direction+"_leg"+str(i+1)+"_joint_z"][1]-=args[i][0]
        else:
            leg_dict[direction+"_leg1"+"_pos"][1]-=args[0][0]
            leg_dict[direction+"_leg2"+"_pos"][1]=leg_dict[direction+"_leg2"+"_pos"][1]-args[0][0]-args[1][0]
            leg_dict[direction+"_leg3"+"_pos"][1]=leg_dict[direction+"_leg3"+"_pos"][1]-args[1][0]-args[2][0]
            leg_dict[direction+"_motor_pos"][1]-=args[2][0]
            for i in range(3):
                leg_dict[direction+"_leg"+str(i+1)+"_joint_x"][1]+=args[i][0]
                leg_dict[direction+"_leg"+str(i+1)+"_joint_z"][1]+=args[i][0]
        
    def muscle_transform(self,leg_n,args):
        leg=["lf","rf","lb","rb"]
        direction=leg[leg_n]
        for i in range(3):
            self.muscle_site[direction+str(i+1)+"_1e"].pos[2]+=args[i][1]
            self.muscle_site[direction+str(i+1)+"_2e"].pos[0]-=args[i][1]
            self.muscle_site[direction+str(i+1)+"_3e"].pos[2]-=args[i][1]
            self.muscle_site[direction+str(i+1)+"_4e"].pos[0]+=args[i][1]
        for i in range(2):
            self.muscle_site[direction+str(i+2)+"_1"].pos[2]+=args[i][1]
            self.muscle_site[direction+str(i+2)+"_2"].pos[0]-=args[i][1]
            self.muscle_site[direction+str(i+2)+"_3"].pos[2]-=args[i][1]
            self.muscle_site[direction+str(i+2)+"_4"].pos[0]+=args[i][1]

    def save(self):
        root=self.mjcf_model.to_xml(precision=3)
        self.indent(root)
        tree=etree.ElementTree(root)
        time=str(datetime.now()).replace(" ","_").split(".")
        name="./robot_xml/make/"+self.name.split(".")[0]+"_"+time[0]+time[1]+".xml"
        tree.write(name)
        return name

    def to_xml_string(self,is_save=False):
        xml_string=self.mjcf_model.to_xml_string()
        if is_save:
            self.save()
        self.reset()
        return xml_string


    def indent(self,elem, level=0):
        i = "\n" + level*"\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
              elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
              elem.tail = i
            for elem in elem:
              self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
             elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
             elem.tail = i




if __name__=="__main__":
    robot=robot_transform("lun_robot.xml")
    args=np.zeros((24))
    for i in range(24):
        args[i]=-1
    robot.transform(args)
    robot.save()
    physics = mjcf.Physics.from_mjcf_model(robot.mjcf_model)
    
    viewer = DMviewer(physics)
    while physics.time() < 1000:
        physics.step()
        viewer.render()
