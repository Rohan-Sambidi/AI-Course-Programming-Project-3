from gen_maze import MazeObject

# Note the color argument overloading in Book and Trolley classes

class DAEBounding(MazeObject):
    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None,name='dae_bounding',id=0,static=True):
        super(DAEBounding, self).__init__(coords=coords,
                         grid_dims=grid_dims,
                         scale=scale,
                         out_xml=out_xml,
                         name=name,
                         dae=dae,
                         id=id,
                         static=static)

    def add_object(self):
        super(DAEBounding, self).add_object()

    def add_object_description(self):
        super(DAEBounding, self).add_object_description()

class Wall(MazeObject):
    '''
    Pose coordinates to set location
    grid size to set length of the wall
    scale to set the <scale> param
    '''

    def __init__(self, coords=None, grid_dims=None, scale=1, out_xml=None, dae=None,id=0, static=True):
        self.x, self.y, self.z = coords
        self.scale = scale[0]
        self.grid_dim = grid_dims[0]
        self.length = self.scale * self.grid_dim
        self.f_out = out_xml

    def add_object(self):
        scale = (self.length + 2) / 7.5
        wall_dimensions = [(-1, self.length / 2, -1.55905, scale, 1),
                           (self.length / 2, self.length + 1, 0, scale, 1),
                           (self.length + 1, self.length / 2, -1.55905, scale, 1),
                           (self.length / 2, -1, 0, scale, 1)]

        for i in range(4):
            self.f_out.write('<model name=\'wall{}\'>\n'.format(i + 1))
            self.f_out.write(
                '<pose frame=\'\'>{} {} 0 0 -0 {}</pose>\n'.format(wall_dimensions[i][0], wall_dimensions[i][1],
                                                                   wall_dimensions[i][2]))
            self.f_out.write('<scale>{} {} 0.03</scale>\n'.format(wall_dimensions[i][3], wall_dimensions[i][4]))
            self.f_out.write('<link name=\'link\'>\n')
            self.f_out.write(
                '<pose frame=\'\'>{} {} 0.42 0 -0 {}</pose>\n'.format(wall_dimensions[i][0], wall_dimensions[i][1],
                                                                      wall_dimensions[i][2]))
            self.f_out.write(
                '<velocity>0 0 0 0 -0 0</velocity>\n<acceleration>0 0 0 0 -0 0</acceleration>\n<wrench>0 0 0 0 -0 0</wrench>\n</link>\n</model>\n')

    def add_object_description(self):
        for i in range(4):
            self.f_out.write('<model name=\'wall{}\'>\n'.format(i + 1))
            self.f_out.write(
                '<static>1</static>\n<link name=\'link\'>\n<pose frame=\'\'>0 0 0.42 0 -0 0</pose>\n<collision name=\'collision\'>\n<geometry>\n<box>\n<size>7.5 0.2 2.8</size>\n</box>\n')
            self.f_out.write(
                '</geometry>\n<max_contacts>10</max_contacts>\n<surface>\n<contact>\n<ode/>\n</contact>\n<bounce/>\n<friction>\n<torsional>\n<ode/>\n</torsional>\n<ode/>\n</friction>\n</surface>\n</collision>\n')
            self.f_out.write(
                '<visual name=\'visual\'>\n<cast_shadows>0</cast_shadows>\n<geometry>\n<box>\n<size>7.5 0.2 2.8</size>\n</box>\n</geometry>\n<material>\n<script>\n')
            self.f_out.write(
                '<uri>model://grey_wall/materials/scripts</uri>\n<uri>model://grey_wall/materials/textures</uri>\n<name>vrc/grey_wall</name>\n</script>\n</material>\n</visual>\n<self_collide>0</self_collide>\n')
            self.f_out.write(
                '<kinematic>0</kinematic>\n<gravity>1</gravity>\n</link>\n<pose frame=\'\'>-0.779308 4.01849 0 0 -0 0</pose>\n</model>\n')

class Table(MazeObject):
    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None,name='table',color=None,id=0,static=True):
        super(Table, self).__init__(coords=coords,
                         grid_dims=grid_dims,
                         scale=scale,
                         out_xml=out_xml,
                         name=name,
                         dae=dae,
                         id=id,
                         static=static)
    def add_object(self):
        super(Table, self).add_object()

    def add_object_description(self):
        
        self.f_out.write("<model name='{}_{}'><pose>0 0 0 0 0 0</pose><static>1</static>\n".format(self.name, self.id))
        self.f_out.write("<link name='link'><collision name='surface'><pose>0 0 0.302 0 0 0</pose><geometry>\n")
        self.f_out.write("<box><size>0.913 0.913 0.04</size></box></geometry></collision>\n")
        self.f_out.write("<collision name='column'><pose>0 0 0.148 0 0 0</pose><geometry><box><size>0.042 0.042 0.74</size></box></geometry></collision>\n")
        self.f_out.write("<collision name='base'><pose>0 0 0.008 0 0 0</pose><geometry><box><size>0.56 0.56 0.04</size></box></geometry></collision>\n")
        self.f_out.write("<visual name='visual'><geometry><mesh><uri>file://{}</uri>\n".format(self.dae))
        self.f_out.write("<scale> {} {} {} </scale></mesh></geometry></visual></link></model>\n".format(self.scale[0],self.scale[1],self.scale[2]))

class Croissant(MazeObject):
    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None,name='croissant',color=None,id=0,static=False):
        super(Croissant, self).__init__(coords=coords,
                         grid_dims=grid_dims,
                         scale=scale,
                         out_xml=out_xml,
                         name=name,
                         dae=dae,
                         id=id,
                         static=static)

    def add_object(self):
        super(Croissant, self).add_object()

    def add_object_description(self):
        super(Croissant, self).add_object_description()

class Hamburger(MazeObject):
    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None,name='hamburger',color=None,id=0,static=False):
        self.name = name
        self.id = id
        self.scale = scale
        self.dae = dae
        super(Hamburger, self).__init__(coords=coords,
                         grid_dims=grid_dims,
                         scale=scale,
                         out_xml=out_xml,
                         name=name,
                         dae=dae,
                         id=id,
                         static=static)

    def add_object(self):
        super(Hamburger, self).add_object()

    def add_object_description(self):
        
        pose_z = self.scale[2]*0.04
        self.f_out.write("<model name='{}_{}'><pose>0 0 0 0 0 0</pose><static>0</static>\n".format(self.name, self.id))
        self.f_out.write("<link name='link'>\n<inertial>\n<mass>0.005</mass>\n")
        self.f_out.write("<inertia>\n<ixx>0.00058</ixx>\n<ixy>0</ixy>\n<ixz>0</ixz>\n<iyy>0.00058</iyy>\n<iyz>0</iyz>\n<izz>0.00019</izz>\n</inertia>\n")
        self.f_out.write("<pose frame=> 0 0 0 0 -0 0</pose>\n")
        self.f_out.write("</inertial>\n<self_collide>0</self_collide>\n<kinematic>0</kinematic>\n<gravity>1</gravity>\n")
        self.f_out.write("<collision name='body_collide'><pose>0 0 {} 0 0 0</pose><geometry>\n".format(pose_z))
        self.f_out.write("<box><size>0.1013 0.1113 0.08</size></box></geometry>")
        self.f_out.write("\n<surface>\n<contact>\n<ode/>\n</contact>\n<bounce/>\n<friction>")
        self.f_out.write("\n<ode><mu>1000</mu><mu2>1000</mu2></ode>\n</friction>\n</surface>\n</collision>\n")
        self.f_out.write("<visual name='visual'><geometry><mesh><uri>file://{}</uri>\n".format(self.dae))
        self.f_out.write("<scale> {} {} {} </scale></mesh></geometry></visual></link></model>\n".format(self.scale[0],self.scale[1],self.scale[2]))

class Can(MazeObject):

    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None,id=0,static=False):

        self.x, self.y, self.z = coords
        self.f_out = out_xml
        self.scale = [x*y for x,y in zip(scale, (1,0.5,1))]
        self.id = id

    def add_object(self):
        self.f_out.write('<model name=\'can_{}{}\'>\n'.format(self.id))
        self.f_out.write('<pose frame=\'\'>{} {} {} 1e-06 0 -9.5e-05</pose>\n'.format(self.x, self.y, self.z))
        self.f_out.write('<scale>{} {} 1</scale>\n<link name=\'link\'>\n'.format(self.scale[0],self.scale[1]))
        self.f_out.write('<pose frame=\'\'>{} {} {} 1e-06 0 -9.5e-05</pose>\n'.format(self.x, self.y, self.z))
        self.f_out.write('<velocity>0 0 0 0 -0 0</velocity>\n<acceleration>0 0 -9.8 0 -0 0</acceleration>\n<wrench>0 0 -3.822 0 -0 0</wrench>\n</link>\n</model>\n')

    def add_object_description(self):
        self.f_out.write('<model name=\'can_{}\'>\n'.format(self.id))
        self.f_out.write('<link name=\'link\'>\n<pose frame=\'\'>0 0 0.115 0 -0 0</pose>\n<inertial>\n<mass>0.39</mass>\n<inertia>\n<ixx>0.00058</ixx>\n<ixy>0</ixy>\n<ixz>0</ixz>\n<iyy>0.00058</iyy>\n<iyz>0</iyz>\n<izz>0.00019</izz>\n</inertia>\n</inertial>\n<collision name=\'collision\'>\n<geometry>\n<cylinder>\n<radius>0.055</radius>\n<length>0.23</length>\n</cylinder>\n</geometry>\n')
        self.f_out.write('<max_contacts>10</max_contacts>\n<surface>\n<contact>\n<ode/>\n</contact>\n<bounce/>\n<friction>\n<torsional>\n<ode/>\n</torsional>\n<ode/>\n</friction>\n</surface>\n</collision>\n<visual name=\'visual\'>\n<geometry>\n<cylinder>\n<radius>0.055</radius>\n<length>0.23</length>\n</cylinder>\n</geometry>\n<material>\n<script>\n<uri>model://beer/materials/scripts</uri>\n<uri>model://beer/materials/textures</uri>\n<name>Beer/Diffuse</name>\n</script>\n</material>\n</visual>\n')
        self.f_out.write('<self_collide>0</self_collide>\n<kinematic>0</kinematic>\n<gravity>1</gravity>\n</link>\n<pose frame=\'\'>0.888525 -2.58346 0 0 -0 0</pose>\n</model>\n')
        self.f_out.write('<gui fullscreen=\'0\'>\n<camera name=\'user_camera\'>\n<pose frame=\'\'>5 -5 2 0 0.275643 2.35619</pose>\n<view_controller>orbit</view_controller>\n<projection_type>perspective</projection_type>\n</camera>\n</gui>\n')

class Cake(MazeObject):
    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None,name='cake',id=0,static=False):
        self.name = name
        self.id = id
        self.scale = scale
        self.dae = dae
        super(Cake, self).__init__(coords=coords,
                         grid_dims=grid_dims,
                         scale=scale,
                         out_xml=out_xml,
                         name=name,
                         dae=dae,
                         id=id,
                         static=static)

    def add_object(self):
        super(Cake, self).add_object()

    def add_object_description(self):
        super(Cake, self).add_object_description()

class Pizza(MazeObject):
    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None,name='pizza',color=None,id=0,static=False):
        self.name = name
        self.id = id
        self.scale = scale
        self.dae = dae
        super(Pizza, self).__init__(coords=coords,
                         grid_dims=grid_dims,
                         scale=scale,
                         out_xml=out_xml,
                         name=name,
                         dae=dae,
                         id=id,
                         static=static)

    def add_object(self):
        super(Pizza, self).add_object()

    def add_object_description(self):
        
        self.f_out.write("<model name='{}_{}'><pose>0 0 0 0 0 0</pose><static>0</static>\n".format(self.name, self.id))
        self.f_out.write("<link name='link'>\n<inertial>\n<mass>0.005</mass>\n")
        self.f_out.write("<inertia>\n<ixx>0.00058</ixx>\n<ixy>0</ixy>\n<ixz>0</ixz>\n<iyy>0.00058</iyy>\n<iyz>0</iyz>\n<izz>0.00019</izz>\n</inertia>\n")
        self.f_out.write("<pose frame=> 0 0 0 0 -0 0</pose>\n")
        self.f_out.write("</inertial>\n<self_collide>0</self_collide>\n<kinematic>0</kinematic>\n<gravity>1</gravity>\n")
        self.f_out.write("<collision name='body_collide'><pose>0.02 0 0 0 0 0</pose><geometry>\n")
        self.f_out.write("<box><size>0.1813 0.1913 0.02</size></box></geometry>")
        self.f_out.write("\n<surface>\n<contact>\n<ode/>\n</contact>\n<bounce/>\n<friction>")
        self.f_out.write("\n<ode><mu>1000</mu><mu2>1000</mu2></ode>\n</friction>\n</surface>\n</collision>\n")
        self.f_out.write("<visual name='visual'><geometry><mesh><uri>file://{}</uri>\n".format(self.dae))
        self.f_out.write("<scale> {} {} {} </scale></mesh></geometry></visual></link></model>\n".format(self.scale[0],self.scale[1],self.scale[2]))

class Book(MazeObject):

    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None, color=0, name="Book", id=0, static=False):
        self.x, self.y, self.z = coords
        self.f_out = out_xml
        self.scale = scale
        self.color = color
        self.name = name
        self.id = id

    def add_object(self):
        self.f_out.write("<model name='book_{}'>\n".format(self.id))
        self.f_out.write("<pose frame=''>{0} {1} -0.000405 -1e-06 1e-06 0</pose>\n".format(self.x, self.y))
        self.f_out.write("<scale>{0} {0} 1</scale>\n".format(self.scale[0],self.scale[1]))
        self.f_out.write("<link name='cover'>\n")
        self.f_out.write("<pose frame=''>{0} {1} 0.015 -1e-06 1e-06 0</pose>\n".format(self.x, self.y-0.000108))
        self.f_out.write("<velocity>0 0 0 0 -0 0</velocity>\n<acceleration>0.017626 0.011511 -0.205341 -0.7674 1.17508 -0</acceleration>\n<wrench>0.017626 0.011511 -0.205341 0 -0 0</wrench>\n</link>\n<link name='page'>\n")
        self.f_out.write("<pose frame=''>{0} {1} 0.015 0 1e-06 0</pose>\n".format(self.x, self.y+0.000608))
        self.f_out.write("<velocity>0 0 0 0 -0 0</velocity>\n<acceleration>0 0 -9.8 0 -0 0</acceleration>\n<wrench>0 0 -9.8 0 -0 0</wrench>\n</link>\n</model>")

    def add_object_description(self):
        self.f_out.write("<model name='book_{}'>\n".format(self.id))
        self.f_out.write("<link name='cover'>\n<pose frame=''>0 -0.000108 0.015405 0 -0 0</pose>\n<self_collide>0</self_collide>\n<kinematic>0</kinematic>\n")
        self.f_out.write("<gravity>1</gravity>\n<visual name='visual'>\n<geometry>\n<box>\n<size>0.245 0.16 0.03</size>\n</box>\n</geometry>\n<material>\n<script>\n")
        self.f_out.write("<uri>model://book_1/materials/scripts/book_{0}.material</uri>\n<uri>model://book_1/materials/textures/cover{0}.jpg</uri>\n<name>book_{0}</name>".format(self.color+1))
        self.f_out.write("\n</script>\n</material>\n<cast_shadows>1</cast_shadows>\n<transparency>0</transparency>\n</visual>\n<collision name='collision'>\n<laser_retro>0</laser_retro>")
        self.f_out.write("\n<max_contacts>10</max_contacts>\n<geometry>\n<box>\n<size>0.245 0.16 0.03</size>\n</box>\n</geometry>\n<surface>\n<contact>\n<ode/>\n</contact>\n<bounce/>\n<friction>")
        self.f_out.write("\n<ode><mu>1000</mu><mu2>1000</mu2></ode>\n</friction>\n</surface>\n</collision>\n<inertial>\n<inertia>\n<ixx>0.00058</ixx>\n<ixy>0</ixy>\n<ixz>0</ixz>\n<iyy>0.00058</iyy>")
        self.f_out.write("\n<iyz>0</iyz>\n<izz>0.00019</izz>\n</inertia>\n<mass>0.05</mass>\n</inertial>\n</link>\n<link name='page'>\n<pose frame=''>0 0.000608 0.015405 0 -0 0</pose>\n<visual name='visual'>")
        self.f_out.write("\n<pose frame=''>0 0 0 0 -0 0</pose>\n<geometry>\n<box>\n<size>0.24502 0.15862 0.028</size>\n</box>\n</geometry>\n<material>\n<lighting>1</lighting>\n<ambient>1 1 1 1</ambient>\n")
        self.f_out.write("<diffuse>1 1 1 1</diffuse>\n<specular>0.01 0.01 0.01 1</specular>\n<emissive>0 0 0 1</emissive>\n<shader type='vertex'>\n<normal_map>__default__</normal_map>\n</shader>\n</material>")
        self.f_out.write("\n<cast_shadows>1</cast_shadows>\n<transparency>0</transparency>\n</visual>\n<collision name='collision'>\n<laser_retro>0</laser_retro>\n<max_contacts>10</max_contacts>\n")
        self.f_out.write("<pose frame=''>0 0 0 0 -0 0</pose>\n<geometry>\n<box>\n<size>0.245 0.16 0.03</size>\n</box>\n</geometry>\n<surface>\n<contact>\n<ode/>\n</contact>\n<bounce/>\n<friction>\n<ode><mu>1000</mu>")
        self.f_out.write("\n<mu2>1000</mu2>\n</ode>\n</friction>\n</surface>\n</collision>\n<self_collide>0</self_collide>\n<inertial>\n<inertia>\n<ixx>0.00058</ixx>\n<ixy>0</ixy>\n<ixz>0</ixz>\n<iyy>0.00058</iyy>\n")
        self.f_out.write("<iyz>0</iyz>\n<izz>0.00019</izz>\n</inertia>\n<mass>0.05</mass>\n</inertial>\n<kinematic>0</kinematic>\n<gravity>1</gravity>\n</link>\n<static>0</static>\n<allow_auto_disable>1</allow_auto_disable>")
        self.f_out.write("\n<pose frame=''>0.830691 0.858956 0 0 -0 0</pose>\n</model>")
        self.f_out.write('<gui fullscreen=\'0\'>\n<camera name=\'user_camera\'>\n<pose frame=\'\'>5 -5 2 0 0.275643 2.35619</pose>\n<view_controller>orbit</view_controller>\n<projection_type>perspective</projection_type>\n</camera>\n</gui>\n')

class Trolley(MazeObject):

    def __init__(self, coords=None, grid_dims=None, scale=(1,1,1), out_xml=None, dae=None, color=(0.0,0.0,0.0), name="Trolley", id=0, static=True):
        self.x, self.y, self.z = coords
        self.f_out = out_xml
        self.scale = scale
        self.color = color
        self.name = name
        self.id = id

    def add_object(self):
        self.f_out.write("<model name='bin_{}'>\n".format(self.id))
        self.f_out.write("<pose frame=''>{0} {1} 0 0 -0 0</pose>".format(self.x, self.y))
        self.f_out.write("<scale>{} {} 0.35</scale>".format(self.scale[0], self.scale[1]))
        self.f_out.write("<link name='link_20'>")
        self.f_out.write("<pose frame=''>{0} {1} 0 0 -0 0</pose>".format(self.x, self.y))
        self.f_out.write("<velocity>0 0 0 0 -0 0</velocity>")
        self.f_out.write("<acceleration>0 0 0 0 -0 0</acceleration>")
        self.f_out.write("<wrench>0 0 0 0 -0 0</wrench>")
        self.f_out.write("</link>")
        self.f_out.write("</model>")


    def add_object_description(self):
        self.f_out.write("<model name='bin_{}'>\n".format(self.id))
        self.f_out.write("<link name='link'>\n<pose frame=''>0 0 0 0 -0 0</pose>\n<inertial>\n<mass>1</mass>\n")
        self.f_out.write("<pose frame=''>0 0 0 0 -0 0</pose>\n<inertia>\n<ixx>1</ixx>\n<ixy>0</ixy>\n<ixz>0</ixz>\n<iyy>1</iyy>\n<iyz>0</iyz>\n<izz>1</izz>\n</inertia>\n</inertial>\n")
        self.f_out.write("<self_collide>0</self_collide>\n<kinematic>0</kinematic>\n<gravity>1</gravity>\n<visual name='visual'>\n<geometry>\n")
        self.f_out.write("<mesh>\n<uri>model://bookcart/meshes/bookcart.dae</uri>\n<scale> {} {} 1</scale>\n</mesh>\n</geometry>\n<pose frame=''>0 0 0 0 -0 0</pose>\n".format(self.scale[0], self.scale[1]))
        self.f_out.write("<cast_shadows>1</cast_shadows>\n<transparency>0</transparency>\n")
        self.f_out.write("<material>\n<ambient>{0} {1} {2} 0</ambient>\n<diffuse>{0} {1} {2} 0</diffuse>\n<specular>0 0 0 1</specular>\n<emissive>0 0 0 1</emissive>\n".format(self.color[0],self.color[1],self.color[2]))
        self.f_out.write("<script>\n<name>ModelPreview_1::link::visual_MATERIAL_</name>\n")
        self.f_out.write("<uri>__default__</uri>\n</script>\n<shader type='vertex'>\n<normal_map>__default__</normal_map>")
        self.f_out.write("\n</shader>\n</material>\n</visual>\n<collision name='collision_0'>\n<laser_retro>0</laser_retro>\n<max_contacts>10</max_contacts>")
        self.f_out.write("\n<pose frame=''>0 0 0 0 -0 0</pose>\n<geometry>\n<mesh>\n<uri>model://bookcart/meshes/bookcart.dae</uri>\n<scale>{} {} 1</scale>\n</mesh>\n</geometry>".format(self.scale[0],self.scale[1]))
        self.f_out.write("\n<surface>\n<friction>\n<ode>\n<mu>1</mu>\n<mu2>1</mu2>\n<fdir1>0 0 0</fdir1>\n<slip1>0</slip1>\n<slip2>0</slip2>\n</ode>\n<torsional>\n<coefficient>1</coefficient>")
        self.f_out.write("\n<patch_radius>0</patch_radius>\n<surface_radius>0</surface_radius>\n<use_patch_radius>1</use_patch_radius>\n<ode>\n<slip>0</slip>\n</ode>")
        self.f_out.write("\n</torsional>\n</friction>\n<bounce>\n<restitution_coefficient>0</restitution_coefficient>\n<threshold>1e+06</threshold>\n</bounce>")
        self.f_out.write("\n<contact>\n<collide_without_contact>0</collide_without_contact>\n<collide_without_contact_bitmask>1</collide_without_contact_bitmask>")
        self.f_out.write("\n<collide_bitmask>1</collide_bitmask>\n<ode>\n")
        self.f_out.write("<soft_cfm>0</soft_cfm>\n<soft_erp>0.2</soft_erp>\n<kp>1e+13</kp>\n<kd>1</kd>\n<max_vel>0.01</max_vel>\n<min_depth>0</min_depth>\n</ode>")
        self.f_out.write("\n<bullet>\n<split_impulse>1</split_impulse>\n<split_impulse_penetration_threshold>-0.01</split_impulse_penetration_threshold>\n<soft_cfm>0</soft_cfm>")
        self.f_out.write("\n<soft_erp>0.2</soft_erp>\n<kp>1e+13</kp>\n<kd>1</kd>\n</bullet>\n</contact>\n</surface>\n</collision>\n</link>\n<static>1</static>\n<allow_auto_disable>1</allow_auto_disable>")
        self.f_out.write("\n<pose frame=''>1.06777 -0.068202 0 0 -0 0</pose>\n</model>")
