import numpy as np
import scipy as sp
import pymech.exadata as exdat
import copy
import matplotlib.path as mpltPath
from pymech.log import logger
from scipy import optimize
#==============================================================================
def extrude(mesh, zlist, bc1, bc2):
    """Extrudes a 2D mesh into a 3D one, given the z position of the nodes.
    
    Parameters
    ----------
    mesh : exadata
           2D mesh structure to extrude
    zlist : float
            list of z values of the nodes of the elements of the extruded mesh
    bc : str
         the boundary condition to use at the ends
    
    Suggestion: you can use define_z to define more complex z positions
    """
    
    #Consistency checks: Initial grid
    if mesh.ndim != 2:
        logger.critical('The mesh to extrude must be 2D')
        return -1
    if mesh.lr1 != [2, 2, 1]:
        logger.critical('Only mesh structures can be extruded (lr1 = [2, 2, 1])')
        return -2
    if mesh.var[0] < 2:
        logger.critical('The mesh to extrude must contain (x, y) geometry')
        return -3
    if (bc1 == 'P' and bc2 != 'P') or (bc1 != 'P' and bc2 == 'P'):
        logger.critical('Inconsistent boundary conditions: one end is \'P\' but the other isn\'t')
        return -4
    
    nz = len(zlist)-1
    z1 = np.zeros((nz,1))
    z2 = np.zeros((nz,1))
    z1 = zlist[0:nz]
    z2 = zlist[1:nz+1]
    
    if (nz<=2 and bc1 == 'P'):
        print('Warning: Nek5000 has experienced errors in the past for n <= 2.') #version 17 of Nek5000 has had errors (Vanishing Jacobian errors) for 2 or less elements in the z direction and periodic boundary condition
    
    # copy the structure and make it 3D
    mesh3d = copy.deepcopy(mesh)
    mesh3d.lr1 = [2, 2, 2]
    mesh3d.var = [3, 0, 0, 0, 0]  # remove anything that isn't geometry for now
    nel2d = mesh.nel
    nel3d = mesh.nel*nz
    nbc = mesh.nbc
    mesh3d.nel = nel3d
    mesh3d.ndim = 3
    # The curved sides will also be extruded, one on each side of each element along nz
    mesh3d.ncurv = 2*nz*mesh.ncurv
    
    # add extra copies of all elements
    for k in range(nz-1):
        mesh3d.elem = mesh3d.elem + copy.deepcopy(mesh.elem)
    
    # set the z locations and curvature
    for k in range(nz):
        for i in range(nel2d):
            iel = i + nel2d*k
            # replace the position arrays with 3D ones (with z empty)
            mesh3d.elem[iel].pos = np.zeros((3, 2, 2, 2))
            mesh3d.elem[iel].pos[:, :1, :, :] = mesh.elem[i].pos
            mesh3d.elem[iel].pos[:, 1:2, :, :] = mesh.elem[i].pos

            # fill in the z location
            mesh3d.elem[iel].pos[2, 0, :, :] = z1[k]
            mesh3d.elem[iel].pos[2, 1, :, :] = z2[k]

            # extend curvature and correct it if necessary
            for icurv in range(4):
                curv_type = mesh3d.elem[iel].ccurv[icurv]
                # a 2D element has 4 edges that can be curved, numbered 0-3;
                # the extruded 3D element can have four more (on the other side), numbered 4-7
                mesh3d.elem[iel].ccurv[icurv+4] = curv_type
                mesh3d.elem[iel].curv[icurv+4] = mesh3d.elem[iel].curv[icurv]    # curvature params
                if curv_type == 'm':
                    # x and y are correct but z should be set to the proper value.
                    mesh3d.elem[iel].curv[icurv][2] = z1[k]
                    mesh3d.elem[iel].curv[icurv+4][2] = z2[k]
                elif curv_type == 's':
                    mesh3d.elem[iel].curv[icurv][2] = z1[k]
                    mesh3d.elem[iel].curv[icurv+4][2] = z2[k]
    
    # fix the internal boundary conditions (even though it's probably useless)
    # the end boundary conditions will be overwritten later with the proper ones
    for (iel, el) in enumerate(mesh3d.elem):
        for ibc in range(nbc):
            # set the conditions in faces normal to z
            el.bcs[ibc, 4][0] = 'E'
            el.bcs[ibc, 4][1] = iel+1
            el.bcs[ibc, 4][2] = 5
            el.bcs[ibc, 4][3] = iel-nel2d+1
            el.bcs[ibc, 4][4] = 6
            el.bcs[ibc, 5][0] = 'E'
            el.bcs[ibc, 5][1] = iel+1
            el.bcs[ibc, 5][2] = 6
            el.bcs[ibc, 5][3] = iel+nel2d+1
            el.bcs[ibc, 5][4] = 5
            # update the conditions for side faces
            for iface in range(4):
                el.bcs[ibc, iface][1] = iel+1
                if (el.bcs[ibc, iface][0] == 'E'):
                    # el.bcs[ibc, 0][1] ought to contain iel+1 once the mesh is valid
                    # but for now it should be off by a factor of nel2d because it is a copy of an element in the first slice
                    offset = iel-el.bcs[ibc, iface][1]+1
                    el.bcs[ibc, iface][3] = el.bcs[ibc, iface][3]+offset
    
    # now fix the end boundary conditions
    # face 5 is at zmin and face 6 is at zmax (with Nek indexing, corresponding to 4 and 5 in Python)
    for i in range(nel2d):
        for ibc in range(nbc):
            i1 = i+(nz-1)*nel2d  # index of the face on the zmax side
            mesh3d.elem[i].bcs[ibc, 4][0] = bc1
            mesh3d.elem[i].bcs[ibc, 4][1] = i+1
            mesh3d.elem[i].bcs[ibc, 4][2] = 5
            mesh3d.elem[i].bcs[ibc, 4][3] = 0.0
            mesh3d.elem[i].bcs[ibc, 4][4] = 0.0
            mesh3d.elem[i1].bcs[ibc, 5][0] = bc2
            mesh3d.elem[i1].bcs[ibc, 5][1] = i1+1
            mesh3d.elem[i1].bcs[ibc, 5][2] = 6
            mesh3d.elem[i1].bcs[ibc, 5][3] = 0.0
            mesh3d.elem[i1].bcs[ibc, 5][4] = 0.0
            # fix the matching faces for the periodic conditions
            if bc1 == 'P':
                mesh3d.elem[i].bcs[ibc, 4][3] = i1+1
                mesh3d.elem[i].bcs[ibc, 4][4] = 6
            if bc2 == 'P':
                mesh3d.elem[i1].bcs[ibc, 5][3] = i+1
                mesh3d.elem[i1].bcs[ibc, 5][4] = 5
    
    # return the extruded mesh
    return mesh3d



#==============================================================================
def extrude_split(mesh2D, zlist, bc1, bc2, fun  = '', funpar = '', imesh_high = 0, verbose = 2):
    """Extrudes a 2D mesh into a 3D one, following the pattern:
     _____ _____ _____ _____
    |     |     |     |     |
    |     |     |     |     |
    |     |     |     |     |
    |_____|_____|_____|_____|
    |    /|\    |    /|\    |
    |__ / | \ __|__ / | \ __| (fun (with parameter funpar) should change change sign in the mid element)
    |  |  |  |  |  |  |  |  | (half of the mid elements are also divided in 2 in (x,y)-plane)
    |__|__|__|__|__|__|__|__|
    |  |  |  |  |  |  |  |  |
    |  |  |  |  |  |  |  |  |
    |  |  |  |  |  |  |  |  | (imesh_high is the index of mesh with higher intended discretization in z)
    |__|__|__|__|__|__|__|__|
    
    The pattern is similar to "Picture Frame" of an ancient NEKTON manual (https://www.mcs.anl.gov/~fischer/Nek5000/nekmanual.pdf).
    If the mid elements have curvature, the extrusion might modify it. Do not split in regions where the value of curvature parameters is very important.
    
    Parameters
    ----------
    mesh2D : exadata
           2D mesh structure to extrude
    zlist : list of floats
            list of z values of the nodes of the elements of the extruded mesh with higher discretization
    bc : str
         the boundary condition to use at the ends
    fun: list of functions
         list of functions that define the splitting lines for different discretization meshes (default: empty, calls simple extrusion routine)
    funpar: list
          list of parameters for functions that define the splitting lines for different discretization meshes (default: empty, for when funpar is not needed inside fun)
    imesh_high : int
                 index of fun that defines the mesh with higher discretization. Example: 0, is the most internal mesh; 1 is the second most internal mesh, etc (default: the most internal mesh, imesh_high=0)
    verbose : int
              level of verbosity. 0: only error and warning messages; 1: number of elements in meshes; 2: all.
    
    Special cases
    -------------
    1) fun = '' : The code ignores the splitting part and calls simple extrusion routine. Observe 'return mesh3D' right after first consistency checks.
    Suggestion: you can use define_z to define more complex z positions
    """
    
    #Consistency checks: Initial grid
    if mesh2D.ndim != 2:
        logger.critical('The mesh to extrude must be 2D')
        return -1
    if mesh2D.lr1 != [2, 2, 1]:
        logger.critical('Only mesh structures can be extruded (lr1 = [2, 2, 1])')
        return -2
    if mesh2D.var[0] < 2:
        logger.critical('The mesh to extrude must contain (x, y) geometry')
        return -3
    #Consistency checks: Periodic boundary condition
    if (bc1 == 'P' and bc2 != 'P') or (bc1 != 'P' and bc2 == 'P'):
        logger.critical('Inconsistent boundary conditions: one end is \'P\' but the other isn\'t')
        return -4
    
    #Consistency checks: Functions that define the splitting lines
    nsplit=len(fun)
    if len(funpar) < nsplit:
        print('Warning: Length of funpar < lenght of fun. Completing with', (nsplit-len(funpar)), 'zeros.')
        funpar[len(funpar):nsplit]=[0.0]*(nsplit-len(funpar))
    elif len(funpar) > nsplit:
        print('Warning: Length of funpar > lenght of fun. Ignoring', (len(funpar)-nsplit), 'values.')
    
    nz = len(zlist)-1
    #Consistency checks: if nz is divided by 4 (or 8, or 16, etc)
    if (nz % 2**abs(imesh_high+1) != 0) or (nz % 2**abs(nsplit-imesh_high+1) != 0):
        logger.critical(f'Inconsistent elements to extrude: n must be divided by {max([2**abs(imesh_high+1),2**abs(nsplit-imesh_high+1)])}')
        return -10
    
    #If fun is not defined, there is no splitting to be done. Call simple extrusion and end routine
    if fun == '':
        if verbose > 1:
            print('Splitting function not defined. Calling simple extrusion routine.')
        mesh3D = extrude(mesh2D, zlist, bc1, bc2)
        return mesh3D
    
    mesh2D_ext = copy.deepcopy(mesh2D)
    meshes2D = [] #list of 2D meshes
    meshes3D = [] #list of 3D meshes
    
    #Splitting 2D meshes
    if verbose > 1:
        print('Splitting 2D meshes. Make sure splitting is done where value of curvature is not important.')
    for k in range(nsplit):
        meshes2D.append(copy.deepcopy(mesh2D_ext))
        meshes2D.append(copy.deepcopy(mesh2D_ext))
        
        iel_int = 0
        iel_mid = 0
        iel_ext = 0
        
        for iel in range(mesh2D_ext.nel):
            it = 0
            xvec = np.zeros((4,1))
            yvec = np.zeros((4,1))
            rvec = np.zeros((4,1))
            for i in range(2):
                for j in range(2):
                    xvec[it] = mesh2D_ext.elem[iel].pos[0,0,j,i]
                    yvec[it] = mesh2D_ext.elem[iel].pos[1,0,j,i]
                    rvec[it] = fun[k](xvec[it], yvec[it], funpar[k])
                    it += 1
            if max(rvec) <= 0.0:
                meshes2D[2*k].elem[iel_int] = copy.deepcopy(mesh2D_ext.elem[iel])
                iel_int += 1
            elif min(rvec) > 0.0:
                mesh2D_ext.elem[iel_ext] = copy.deepcopy(mesh2D_ext.elem[iel])
                iel_ext += 1
            else:
                meshes2D[2*k+1].elem[iel_mid] = copy.deepcopy(mesh2D_ext.elem[iel])
                iel_mid += 1
                        
        meshes2D[2*k].nel=iel_int
        meshes2D[2*k+1].nel=iel_mid
        mesh2D_ext.nel=iel_ext
        
        meshes2D[2*k].elem=meshes2D[2*k].elem[:iel_int]
        meshes2D[2*k+1].elem=meshes2D[2*k+1].elem[:iel_mid]
        mesh2D_ext.elem=mesh2D_ext.elem[:iel_ext]
        
        update_ncurv(meshes2D[2*k])
        update_ncurv(meshes2D[2*k+1])
        
        if verbose > 0:
            print('Mesh2D', 2*k, 'elements:', meshes2D[2*k].nel)
            print('Mesh2D', 2*k+1, 'elements:', meshes2D[2*k+1].nel)
    
    #End of splitting, remaining is the last mesh: mesh2D_ext
    update_ncurv(mesh2D_ext)
    if verbose > 0:
        print('Mesh2Dext elements:', mesh2D_ext.nel)
    
    #Extruding meshes
    if verbose > 1:
        print('Extruding meshes')
    for k in range(nsplit):
        zlist_local=zlist[::int(2**abs(k-imesh_high))]
        
        if k<imesh_high:
            fun_local = lambda xpos, ypos, rlim : -fun[k](xpos,ypos,rlim)
            zlist_mid=zlist[::int(2**abs(k-imesh_high+1))]
        else:
            fun_local = fun[k]
            zlist_mid=zlist_local
        
        meshes3D.append(extrude(meshes2D[2*k], zlist_local, bc1, bc2))
        meshes3D.append(extrude_mid(meshes2D[2*k+1], zlist_mid, bc1, bc2, fun_local, funpar[k]))
        
        #print('Removing boundary condition E to improve merging time')
        delete_internal_bcs(meshes3D[2*k])
        delete_internal_bcs(meshes3D[2*k+1])
        
        if verbose > 0:
            print('Mesh3D', 2*k, 'elements:', meshes3D[2*k].nel)
            print('Mesh3D', 2*k+1, 'elements:', meshes3D[2*k+1].nel)
    
    zlist_local = zlist[::int(2**abs(nsplit-imesh_high))]
    mesh3D_ext = extrude(mesh2D_ext, zlist_local, bc1, bc2)
    delete_internal_bcs(mesh3D_ext)
    
    if verbose > 0:
        print('Mesh3Dext elements:', mesh3D_ext.nel)
    
    #Merging meshes
    if verbose > 1:
        print('Merging meshes')
        print('Boundary condition E were removed to improve merging time')
    for k in range(nsplit):
        if k==0:
            mesh3D = copy.deepcopy(meshes3D[2*k])
        else:
            mesh3D = merge(mesh3D,meshes3D[2*k])
        #print('Mesh', 2*k, 'merged. Elements:', mesh3D.nel)
        mesh3D = merge(mesh3D,meshes3D[2*k+1])
        #print('Mesh', 2*k+1, 'merged. Elements:', mesh3D.nel)
    
    mesh3D = merge(mesh3D,mesh3D_ext)
    #print('Mesh ext merged. Elements:', mesh3D.nel)
    if verbose > 0:
        print('Merging done. Total elements:', mesh3D.nel)
    
    # Count the curved elements (should be correct, but just to be sure)
    update_ncurv(mesh3D)
    
    # return the extruded mesh
    return mesh3D



#==============================================================================
def extrude_mid(mesh, zlist, bc1, bc2, fun, funpar = 0.0):
    """Extrudes the mid elments of the 2D mesh into a 3D one. Following the pattern:
     _____ _____ _____ _____ 
    |1   /|\   4|    /|\    |
    |__ / | \ __|__ / | \ __| (fun (with parameter funpar) should change change sign in the mid element)
    |0 |2 |3 | 5|  |  |  |  | (half of the mid elements are also divided in 2 in (x,y)-plane)
    |__|__|__|__|__|__|__|__| (numbers in the figure indicate the indices (iel+0; iel+1; etc))
    
    Parameters
    ----------
    mesh : exadata
           2D mesh structure to extrude
    zlist : float
            list of z values of the nodes of the elements of the extruded mesh in the high discretization region (len(zlist)-1 must be divide by 4)
    bc : str
         the boundary condition to use at the ends
    fun : function
          function that define the splitting lines for different discretization meshes
    funpar : not defined, depends on the function
             parameter for functions that define the splitting lines for different discretization meshes (default: zero, can be used for when funpar is not needed inside fun)

    Suggestion: see function extrude_split to understand how to call extrude_mid
    """
    
    #Consistency checks: Initial grid
    if mesh.ndim != 2:
        logger.critical('The mesh to extrude must be 2D')
        return -1
    if mesh.lr1 != [2, 2, 1]:
        logger.critical('Only mesh structures can be extruded (lr1 = [2, 2, 1])')
        return -2
    if mesh.var[0] < 2:
        logger.critical('The mesh to extrude must contain (x, y) geometry')
        return -3
    if (bc1 == 'P' and bc2 != 'P') or (bc1 != 'P' and bc2 == 'P'):
        logger.critical('Inconsistent boundary conditions: one end is \'P\' but the other isn\'t')
        return -4
    
    nz=len(zlist)-1
    z1 = np.zeros((nz,1))
    z2 = np.zeros((nz,1))
    z1 = zlist[0:nz]
    z2 = zlist[1:nz+1]
    
    if nz % 4 != 0:
        logger.critical('Inconsistent elements to extrude: nz must be divided by 4')
        return -5
 
    # copy the structure and make it 3D
    mesh3d = copy.deepcopy(mesh)
    mesh3d.lr1 = [2, 2, 2]
    mesh3d.var = [3, 0, 0, 0, 0]  # remove anything that isn't geometry for now
    nel2d = mesh.nel
    nel3d = nel2d*6*int(nz/4) # every mid-extrusion of a 2d-element creates 6 3d-elements, while the usual extrusion creates 4 in the high discretized grid and 2 in the low discretized grid
    nbc = mesh.nbc
    mesh3d.nel = nel3d
    mesh3d.ndim = 3
    # The curved sides will also be extruded, one on each side of each element along nz
    mesh3d.ncurv = 2*nz*mesh.ncurv

    # add extra copies of all elements
    for k in range(6*int(nz/4)-1):
        mesh3d.elem = mesh3d.elem + copy.deepcopy(mesh.elem)

    # set the x,y,z locations and curvature
    for k in range(0, nz, 4):
        for i in range(nel2d):
            iel = 6*(i + nel2d*int(k/4))
            for iell in range(6):
                mesh3d.elem[iel+iell] = copy.deepcopy(mesh.elem[i])
                mesh3d.elem[iel+iell].pos = np.zeros((3, 2, 2, 2))
            
            xvec = np.zeros((2,2))
            yvec = np.zeros((2,2))
            rvec = np.zeros((2,2))
            index_lo = np.zeros((2,2), dtype=int) #index of low points
            index_hi = np.zeros((2,2), dtype=int) #index of high points
            iindex_lo = 0
            iindex_hi = 0
            iedgelat = np.zeros((2), dtype=int)
            iedgeconlat = np.zeros((2), dtype=int)
            for ii in range(2):
                for jj in range(2):
                    xvec[jj,ii] = mesh.elem[i].pos[0,0,jj,ii]
                    yvec[jj,ii] = mesh.elem[i].pos[1,0,jj,ii]
                    rvec[jj,ii] = fun(xvec[jj,ii], yvec[jj,ii], funpar)
                    if rvec[jj,ii] <= 0.0:
                        if (iindex_lo > 1):
                            logger.critical('Mid element not consistent. Criteria must divide elements with 2 points on each side.')
                            return -11
                        index_lo[iindex_lo,:]=[jj,ii]
                        iindex_lo += 1
                    else:
                        if (iindex_hi > 1):
                            logger.critical('Mid element not consistent. Criteria must divide elements with 2 points on each side.')
                            return -11
                        index_hi[iindex_hi,:]=[jj,ii]
                        iindex_hi += 1
            if (iindex_lo != 2) or (iindex_hi != 2):
                logger.critical('Mid element not consistent. Criteria must divide elements with 2 points on each side.')
                return -11
            
            # find the indices of edges, for curvature and boundary condition
            # iedgehi is the index of the edge of element iel+0 that is the intersection between iel+0 and iel+1 (high edge). iedgelo is the index of the edge of element iel+1 that is the intersection (low edge).
            # iedgelat are the indices of the lateral (splitted) edges. iedgeconlat are the indices of the edges (edge in z-direction) of elements iel+2 and iel+3 that connect to the respective lateral edges.
            if (index_lo[0,:]==[0,0]).all():
                if (index_hi[0,:]==[0,1]).all():
                    iedgehi = 1
                    iedgelo = 3
                    iedgelat[0] = 0
                    iedgelat[1] = 2
                    iedgeconlat[0] = 9
                    iedgeconlat[1] = 10
                else:
                    iedgehi = 2
                    iedgelo = 0
                    iedgelat[0] = 3
                    iedgelat[1] = 1
                    iedgeconlat[0] = 11
                    iedgeconlat[1] = 10
            elif (index_lo[0,:]==[1,0]).all():
                iedgehi = 0
                iedgelo = 2
                iedgelat[0] = 3
                iedgelat[1] = 1
                iedgeconlat[0] = 8
                iedgeconlat[1] = 9
            elif (index_lo[0,:]==[0,1]).all():
                iedgehi = 3
                iedgelo = 1
                iedgelat[0] = 0
                iedgelat[1] = 2
                iedgeconlat[0] = 8
                iedgeconlat[1] = 11
            
            # find x and y locations
            poslo = copy.deepcopy(mesh.elem[i].pos[:, :1, index_lo[:,0], index_lo[:,1]])
            poshi = copy.deepcopy(mesh.elem[i].pos[:, :1, index_hi[:,0], index_hi[:,1]])
            # mid position is influenced by curvature
            posmid  = 0.5*(mesh.elem[i].pos[:,:1, index_lo[:,0], index_lo[:,1]] + mesh.elem[i].pos[:,:1, index_hi[:,0], index_hi[:,1]])
            for ilat in range(2):
                #finds the mid points of lateral edges (also considering curvature)
                posmid[:,0,ilat] = edge_mid(mesh.elem[i],iedgelat[ilat])
            
            # fill in the x and y location
            mesh3d.elem[iel].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = poslo
            mesh3d.elem[iel].pos[:, 0:2, index_hi[:,0], index_hi[:,1]] = posmid
            
            mesh3d.elem[iel+1].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = posmid
            mesh3d.elem[iel+1].pos[:, 0:2, index_hi[:,0], index_hi[:,1]] = poshi
            
            mesh3d.elem[iel+2].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = poslo
            mesh3d.elem[iel+2].pos[:,  :1, index_hi[:,0], index_hi[:,1]] = posmid
            mesh3d.elem[iel+2].pos[:, 1:2, index_hi[:,0], index_hi[:,1]] = poshi
            
            mesh3d.elem[iel+3].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = poslo
            mesh3d.elem[iel+3].pos[:,  :1, index_hi[:,0], index_hi[:,1]] = poshi
            mesh3d.elem[iel+3].pos[:, 1:2, index_hi[:,0], index_hi[:,1]] = posmid
            
            mesh3d.elem[iel+4].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = posmid
            mesh3d.elem[iel+4].pos[:, 0:2, index_hi[:,0], index_hi[:,1]] = poshi
            
            mesh3d.elem[iel+5].pos[:, 0:2, index_lo[:,0], index_lo[:,1]] = poslo
            mesh3d.elem[iel+5].pos[:, 0:2, index_hi[:,0], index_hi[:,1]] = posmid

            # fill in the z location
            mesh3d.elem[iel].pos[2, 0, :, :] = z1[k]
            mesh3d.elem[iel].pos[2, 1, :, :] = z2[k]
            
            mesh3d.elem[iel+1].pos[2, 0, :, :] = z1[k]
            mesh3d.elem[iel+1].pos[2, 1, index_lo[0,0], index_lo[0,1]] = z2[k]
            mesh3d.elem[iel+1].pos[2, 1, index_lo[1,0], index_lo[1,1]] = z2[k]
            mesh3d.elem[iel+1].pos[2, 1, index_hi[0,0], index_hi[0,1]] = z2[k+1]
            mesh3d.elem[iel+1].pos[2, 1, index_hi[1,0], index_hi[1,1]] = z2[k+1]
            
            mesh3d.elem[iel+2].pos[2, 0, :, :] = z1[k+1]
            mesh3d.elem[iel+2].pos[2, 1, :, :] = z2[k+1]
            
            mesh3d.elem[iel+3].pos[2, 0, :, :] = z1[k+2]
            mesh3d.elem[iel+3].pos[2, 1, :, :] = z2[k+2]
            
            mesh3d.elem[iel+4].pos[2, 1, :, :] = z2[k+3]
            mesh3d.elem[iel+4].pos[2, 0, index_lo[0,0], index_lo[0,1]] = z1[k+3]
            mesh3d.elem[iel+4].pos[2, 0, index_lo[1,0], index_lo[1,1]] = z1[k+3]
            mesh3d.elem[iel+4].pos[2, 0, index_hi[0,0], index_hi[0,1]] = z1[k+2]
            mesh3d.elem[iel+4].pos[2, 0, index_hi[1,0], index_hi[1,1]] = z1[k+2]
            
            mesh3d.elem[iel+5].pos[2, 0, :, :] = z1[k+3]
            mesh3d.elem[iel+5].pos[2, 1, :, :] = z2[k+3]
            
            for icurv in range(4):
                # Curvature type 's' acts on faces, not edges. Does not make sense for extruded mesh. Changing to 'C'.
                if mesh.elem[i].ccurv[icurv] == 's':
                    print('Curvature s on element', i+1,'. Not consistent with extrusion, changing to C')
                    mesh.elem[i].ccurv[icurv] == 'C'
                    mesh.elem[i].ccurv[icurv][0] = mesh.elem[i].ccurv[icurv][4]
                    mesh.elem[i].ccurv[icurv][1:4] = 0.0
                elif (mesh.elem[i].ccurv[icurv] != '') and (mesh.elem[i].ccurv[icurv] != 'm') and (mesh.elem[i].ccurv[icurv] != 'C'):
                    print('Warning: Curvature unknown on element', i+1,'. Unpredictable behaviour.')
            
            # extend curvature and correct it if necessary
            # calculate coordinates of midsize-node for every curvature type, except both empty (even if both are 'C'). Then find radius, if applicable. 'm' takes precendence over 'C'.
            #if mesh.elem[i].ccurv[iedgehi] != mesh.elem[i].ccurv[iedgelo]:
            if mesh.elem[i].ccurv[iedgehi] != '' or mesh.elem[i].ccurv[iedgelo] != '':
                midpointhi = edge_mid(mesh.elem[i], iedgehi)
                midpointlo = edge_mid(mesh.elem[i], iedgelo)
                midpointmid = 0.5*(posmid[:,0,0]+posmid[:,0,1]) + 0.5*(midpointhi-0.5*(poshi[:,0,0]+poshi[:,0,1])+midpointlo-0.5*(poslo[:,0,0]+poslo[:,0,1]))
                if mesh.elem[i].ccurv[iedgehi]=='m' or mesh.elem[i].ccurv[iedgelo]=='m':
                    mesh3d.elem[iel].ccurv[iedgehi] = 'm'
                    mesh3d.elem[iel+1].ccurv[iedgelo] = 'm'
                    mesh3d.elem[iel].curv[iedgehi][:3] = midpointmid
                    mesh3d.elem[iel+1].curv[iedgelo][:3] = midpointmid
                elif mesh.elem[i].ccurv[iedgehi]=='C' or mesh.elem[i].ccurv[iedgelo]=='C':
                    midpointmid[2] = z1[k]
                    curvmid = edge_circle(mesh3d.elem[iel], iedgehi, midpointmid)
                    if curvmid[0] == 0.0:
                        mesh3d.elem[iel].ccurv[iedgehi] = ''
                        mesh3d.elem[iel+1].ccurv[iedgelo] = ''
                        mesh3d.elem[iel].curv[iedgehi][:4] = 0.0
                        mesh3d.elem[iel+1].curv[iedgelo][:4] = 0.0
                    else:
                        curvmid[1:4] = 0.0
                        mesh3d.elem[iel].ccurv[iedgehi] = 'C'
                        mesh3d.elem[iel+1].ccurv[iedgelo] = 'C'
                        mesh3d.elem[iel].curv[iedgehi][:4] = curvmid
                        mesh3d.elem[iel+1].curv[iedgelo][:4] = -curvmid
                else:
                    print('Warning: Splitted element curvature unknown on element', i+1,'of mid mesh. Removing curvature.')
                    #For cases not implemented, remove curvature
                    mesh3d.elem[iel].ccurv[iedgehi] = ''
                    mesh3d.elem[iel+1].ccurv[iedgelo] = ''
                    mesh3d.elem[iel].curv[iedgehi] = 0.0*(mesh.elem[i].curv[iedgehi])
                    mesh3d.elem[iel+1].curv[iedgelo] = copy.deepcopy(mesh3d.elem[iel].curv[iedgehi])
                
            for ilat in range(2):
                # Fixing curvature of edges divided in half. For curv_type == 'C', it is not a true extrusion - 'diagonal edges' not consistent with 'C'.
                if mesh.elem[i].ccurv[iedgelat[ilat]]=='m':
                    # coordinates of midsize-node is approximated. Considering an edge aligned with the x-axis, the position would be (ym_new = 3/4*ym_old = 1/2*ym_old(mean y-position)+1/4*ym_old(distance)) at xm_new=(x2-x1)/4. This comes from parabolic curve (however, it is not exactly midsize)
                    dmid = ((poshi[0,0,ilat]-poslo[0,0,ilat])*(poslo[1,0,ilat]-posmid[1,0,ilat])-(poslo[0,0,ilat]-posmid[0,0,ilat])*(poshi[1,0,ilat]-poslo[1,0,ilat]))/((poshi[0,0,ilat]-poslo[0,0,ilat])**2+(poshi[1,0,ilat]-poslo[1,0,ilat])**2)
                    mesh3d.elem[iel].curv[iedgelat[ilat]][0]   = 0.5*(poslo[0,0,ilat]+posmid[0,0,ilat]) + dmid/4.0*(poshi[1,0,ilat]-poslo[1,0,ilat])
                    mesh3d.elem[iel+1].curv[iedgelat[ilat]][0] = 0.5*(posmid[0,0,ilat]+poshi[0,0,ilat]) + dmid/4.0*(poshi[1,0,ilat]-poslo[1,0,ilat])
                    mesh3d.elem[iel].curv[iedgelat[ilat]][1]   = 0.5*(poslo[1,0,ilat]+posmid[1,0,ilat]) - dmid/4.0*(poshi[0,0,ilat]-poslo[0,0,ilat])
                    mesh3d.elem[iel+1].curv[iedgelat[ilat]][1] = 0.5*(posmid[1,0,ilat]+poshi[1,0,ilat]) - dmid/4.0*(poshi[0,0,ilat]-poslo[0,0,ilat])
                elif mesh.elem[i].ccurv[iedgelat[ilat]]=='C':
                    # if the lateral edge has curvature 'C', the diagonal edges connected to it (iedgeconlat of elements iel+2 and iel+3) would have curvature 'C', which are inconsistent with edges 8-12 inside Nek5000 (because these are supposed to be edges in z-direction). Changing to curvature 'm' for elements iel+1 (and iel+2, iel+3, iel+2, iel+4)
                    midpointlathi = edge_mid(mesh3d.elem[iel+1], iedgelat[ilat])
                    mesh3d.elem[iel+1].curv[iedgelat[ilat]][:3] = midpointlathi
                    mesh3d.elem[iel+1].ccurv[iedgelat[ilat]] = 'm'
            
            for icurv in range(4):
                # a 2D element has 4 edges that can be curved, numbered 0-3;
                # the extruded 3D element can have four more (on the other side), numbered 4-7
                for iell in range(6):
                    mesh3d.elem[iel+iell].ccurv[icurv+4] = mesh3d.elem[iel+iell].ccurv[icurv]
                    mesh3d.elem[iel+iell].curv[icurv+4] = mesh3d.elem[iel+iell].curv[icurv]    # curvature params
                    
            mesh3d.elem[iel+2].curv[0:4] = copy.deepcopy(mesh3d.elem[iel].curv[0:4])
            mesh3d.elem[iel+3].curv[4:8] = copy.deepcopy(mesh3d.elem[iel].curv[0:4])
            mesh3d.elem[iel+5].curv      = copy.deepcopy(mesh3d.elem[iel].curv)
            mesh3d.elem[iel+4].curv      = copy.deepcopy(mesh3d.elem[iel+1].curv)
            mesh3d.elem[iel+2].ccurv[0:4] = copy.deepcopy(mesh3d.elem[iel].ccurv[0:4])
            mesh3d.elem[iel+3].ccurv[4:8] = copy.deepcopy(mesh3d.elem[iel].ccurv[0:4])
            mesh3d.elem[iel+5].ccurv      = copy.deepcopy(mesh3d.elem[iel].ccurv)
            mesh3d.elem[iel+4].ccurv      = copy.deepcopy(mesh3d.elem[iel+1].ccurv)
            
            for icurv in range(4):
                # z should be set to the proper value.
                curv_type = mesh3d.elem[iel].ccurv[icurv]
                if curv_type == 'm':
                    izcurv=2
                    mesh3d.elem[iel].curv[icurv][izcurv]     = z1[k]
                    mesh3d.elem[iel].curv[icurv+4][izcurv]   = z2[k]
                    mesh3d.elem[iel+2].curv[icurv][izcurv]   = z1[k+1]
                    mesh3d.elem[iel+2].curv[icurv+4][izcurv] = z2[k+1]
                    mesh3d.elem[iel+3].curv[icurv][izcurv]   = z1[k+2]
                    mesh3d.elem[iel+3].curv[icurv+4][izcurv] = z2[k+2]
                    mesh3d.elem[iel+5].curv[icurv][izcurv]   = z1[k+3]
                    mesh3d.elem[iel+5].curv[icurv+4][izcurv] = z2[k+3]
                
                curv_type = mesh3d.elem[iel+1].ccurv[icurv]
                # curvature of iel+1 may be different from iel because of diagonal edges
                if curv_type == 'm':
                    izcurv=2
                    mesh3d.elem[iel+1].curv[icurv][izcurv]   = z1[k]
                    mesh3d.elem[iel+4].curv[icurv+4][izcurv] = z2[k+3]
                    if icurv==iedgehi:
                        mesh3d.elem[iel+1].curv[iedgehi+4][izcurv] = z2[k+1]
                        mesh3d.elem[iel+4].curv[iedgehi][izcurv]   = z2[k+1]
                    elif icurv==iedgelo:
                        mesh3d.elem[iel+1].curv[iedgelo+4][izcurv] = z1[k+1]
                        mesh3d.elem[iel+4].curv[iedgelo][izcurv]   = z2[k+2]
                    elif icurv==iedgelat[0] or icurv==iedgelat[1]:
                        mesh3d.elem[iel+1].curv[icurv+4][izcurv]    = 0.5*(z1[k+1]+z2[k+1])
                        mesh3d.elem[iel+4].curv[icurv][izcurv]      = 0.5*(z2[k+1]+z2[k+2])
            
            #Fixing the curvature of 3d-edges in z-direction that connects to lateral edges in trapezoidal elements (all other edges in z-direction - indices 8 to 11 - do not have curvature)
            for ilat in range(2):
                mesh3d.elem[iel+2].curv[iedgeconlat[ilat]] = copy.deepcopy(mesh3d.elem[iel+1].curv[iedgelat[ilat]+4])
                mesh3d.elem[iel+2].ccurv[iedgeconlat[ilat]] = copy.deepcopy(mesh3d.elem[iel+1].ccurv[iedgelat[ilat]+4])
                mesh3d.elem[iel+3].curv[iedgeconlat[ilat]] = copy.deepcopy(mesh3d.elem[iel+4].curv[iedgelat[ilat]])
                mesh3d.elem[iel+3].ccurv[iedgeconlat[ilat]] = copy.deepcopy(mesh3d.elem[iel+4].ccurv[iedgelat[ilat]])
            
    # fix the internal boundary conditions (even though it's probably useless) 
    # the end boundary conditions will be overwritten later with the proper ones
            for ibc in range(nbc):
                # set the conditions in faces normal to z
                for iell in range(6):
                    mesh3d.elem[iel+iell].bcs[ibc, 4][0] = 'E'
                    mesh3d.elem[iel+iell].bcs[ibc, 4][1] = iel+iell+1
                    mesh3d.elem[iel+iell].bcs[ibc, 4][2] = 5
                    mesh3d.elem[iel+iell].bcs[ibc, 4][4] = 6
                    mesh3d.elem[iel+iell].bcs[ibc, 5][0] = 'E'
                    mesh3d.elem[iel+iell].bcs[ibc, 5][1] = iel+iell+1
                    mesh3d.elem[iel+iell].bcs[ibc, 5][2] = 6
                    mesh3d.elem[iel+iell].bcs[ibc, 5][4] = 5
                
                mesh3d.elem[iel].bcs[ibc, 4][3] = iel+1-nel2d
                mesh3d.elem[iel].bcs[ibc, 5][3] = iel+1+2
                mesh3d.elem[iel+1].bcs[ibc, 4][3] = iel+1-nel2d
                mesh3d.elem[iel+1].bcs[ibc, 5][3] = iel+1+2
                mesh3d.elem[iel+2].bcs[ibc, 4][3] = iel+1
                mesh3d.elem[iel+2].bcs[ibc, 5][3] = iel+1+3
                mesh3d.elem[iel+3].bcs[ibc, 4][3] = iel+1+2
                mesh3d.elem[iel+3].bcs[ibc, 5][3] = iel+1+5
                mesh3d.elem[iel+4].bcs[ibc, 4][3] = iel+1+3
                mesh3d.elem[iel+4].bcs[ibc, 5][3] = iel+1+nel2d
                mesh3d.elem[iel+5].bcs[ibc, 4][3] = iel+1+3
                mesh3d.elem[iel+5].bcs[ibc, 5][3] = iel+1+nel2d
                # update the conditions for side faces. (FIXME : Not corrected for merging - need to know numbering of other simply-extruded meshes (it is not really necessary, internal bc are not used))
                for iface in range(4):
                    for iell in range(6):
                        mesh3d.elem[iel+iell].bcs[ibc, iface][1] = iel+iell+1
                        mesh3d.elem[iel+iell].bcs[ibc, iface][2] = iface+1
                    if mesh3d.elem[iel].bcs[ibc, iface][0] == 'E':
                        # el.bcs[ibc, 0][1] ought to contain iel+1 once the mesh is valid
                        # but for now it should be off by a factor of nel2d because it is a copy of an element in the first slice
                        ielneigh = 6*(mesh3d.elem[iel].bcs[ibc, iface][3]-1 + nel2d*int(k/4))
                        for iell in range(6):
                            mesh3d.elem[iel+iell].bcs[ibc, iface][3] = ielneigh+1+iell
                
                # Correct internal bc for mid faces of elements.
                mesh3d.elem[iel].bcs[ibc, iedgehi][0] = 'E'
                mesh3d.elem[iel].bcs[ibc, iedgehi][3] = iel+1+1
                mesh3d.elem[iel].bcs[ibc, iedgehi][4] = iedgelo+1
                mesh3d.elem[iel+1].bcs[ibc, iedgelo][0] = 'E'
                mesh3d.elem[iel+1].bcs[ibc, iedgelo][3] = iel+1
                mesh3d.elem[iel+1].bcs[ibc, iedgelo][4] = iedgehi+1
                mesh3d.elem[iel+1].bcs[ibc, 5][0] = 'E'
                mesh3d.elem[iel+1].bcs[ibc, 5][3] = iel+1+2
                mesh3d.elem[iel+1].bcs[ibc, 5][4] = iedgehi+1
                mesh3d.elem[iel+2].bcs[ibc, iedgehi][0] = 'E'
                mesh3d.elem[iel+2].bcs[ibc, iedgehi][3] = iel+1+1
                mesh3d.elem[iel+2].bcs[ibc, iedgehi][4] = 6
                mesh3d.elem[iel+3].bcs[ibc, iedgehi][0] = 'E'
                mesh3d.elem[iel+3].bcs[ibc, iedgehi][3] = iel+1+4
                mesh3d.elem[iel+3].bcs[ibc, iedgehi][4] = 5
                mesh3d.elem[iel+4].bcs[ibc, 4][0] = 'E'
                mesh3d.elem[iel+4].bcs[ibc, 4][3] = iel+1+3
                mesh3d.elem[iel+4].bcs[ibc, 4][4] = iedgehi+1
                mesh3d.elem[iel+4].bcs[ibc, iedgelo][0] = 'E'
                mesh3d.elem[iel+4].bcs[ibc, iedgelo][3] = iel+1+5
                mesh3d.elem[iel+4].bcs[ibc, iedgelo][4] = iedgehi+1
                mesh3d.elem[iel+5].bcs[ibc, iedgehi][0] = 'E'
                mesh3d.elem[iel+5].bcs[ibc, iedgehi][3] = iel+1+4
                mesh3d.elem[iel+5].bcs[ibc, iedgehi][4] = iedgelo+1
                
    # now fix the end boundary conditions 
    # face 5 is at zmin and face 6 is at zmax (with Nek indexing, corresponding to 4 and 5 in Python)
    for i in range(0,6*nel2d,6):
        for ibc in range(nbc):
            i1 = i+nel3d-6*nel2d+5
            mesh3d.elem[i].bcs[ibc, 4][0] = bc1
            mesh3d.elem[i].bcs[ibc, 4][1] = i+1
            mesh3d.elem[i].bcs[ibc, 4][2] = 5
            mesh3d.elem[i+1].bcs[ibc, 4][0] = bc1
            mesh3d.elem[i+1].bcs[ibc, 4][1] = i+1+1
            mesh3d.elem[i+1].bcs[ibc, 4][2] = 5
            mesh3d.elem[i1].bcs[ibc, 5][0] = bc2
            mesh3d.elem[i1].bcs[ibc, 5][1] = i1+1
            mesh3d.elem[i1].bcs[ibc, 5][2] = 6
            mesh3d.elem[i1-1].bcs[ibc, 5][0] = bc2
            mesh3d.elem[i1-1].bcs[ibc, 5][1] = i1-1+1
            mesh3d.elem[i1-1].bcs[ibc, 5][2] = 6
            
            mesh3d.elem[i].bcs[ibc, 4][3] = 0.0
            mesh3d.elem[i].bcs[ibc, 4][4] = 0.0
            mesh3d.elem[i+1].bcs[ibc, 4][3] = 0.0
            mesh3d.elem[i+1].bcs[ibc, 4][4] = 0.0
            mesh3d.elem[i1].bcs[ibc, 5][3] = 0.0
            mesh3d.elem[i1].bcs[ibc, 5][4] = 0.0
            mesh3d.elem[i1-1].bcs[ibc, 5][3] = 0.0
            mesh3d.elem[i1-1].bcs[ibc, 5][4] = 0.0
            
            # fix the matching faces for the periodic conditions
            if bc1 == 'P':
                mesh3d.elem[i].bcs[ibc, 4][3] = i1+1
                mesh3d.elem[i].bcs[ibc, 4][4] = 6
                mesh3d.elem[i+1].bcs[ibc, 4][3] = i1-1+1
                mesh3d.elem[i+1].bcs[ibc, 4][4] = 6
            if bc2 == 'P':
                mesh3d.elem[i1].bcs[ibc, 5][3] = i+1
                mesh3d.elem[i1].bcs[ibc, 5][4] = 5
                mesh3d.elem[i1-1].bcs[ibc, 5][3] = i+1+1
                mesh3d.elem[i1-1].bcs[ibc, 5][4] = 5
    
    # Removing internal boundary conditions. 'E' boundary conditions are wrong where meshes merge, but should be right internally. (When possible: FIXME indices and delete these lines. However, it is not really necessary, internal bc are not used)
    delete_internal_bcs(mesh3d)
    
    # Count the curved elements
    update_ncurv(mesh3d)
    
    # return the extruded mesh
    return mesh3d



#==============================================================================
def taper(mesh, ble, bte, dih = 0.0, coord = [''], z0 = 0.0):
    """Modifies a 3D mesh (ideally extruded) to include taper, sweep and dihedral:
    
    Parameters
    ----------
    mesh : exadata
           3D mesh structure to modify
    ble : float
          sweep angle (degrees) at the leading edge. A positive ble bends the mesh inward
    bte : float
          sweep angle (degrees) at the trailing edge. A positive bte bends the mesh inward (notice that it is opposite of aeronautical convention for wings)
    dih : float
          dihedral angle (degrees) at the leading edge. A positive dih bends the mesh upward (default: 0.0, no dihedral)
    coord : float
            array of length four with the structure (xle,yle,xte,yte) where xle are the x and y coordinates of the leading edge. xte and yte are the counterparts for the trailing edge (default: [''], minimum/maximum x and y values - see special case 1)
    z0 : float
         reference z-position around which the taper, sweep and dihedral will be performed, the position which the mesh will not be scaled/deformed (default: 0.0)

    Special cases
    -------------
    1) coord = [''] : xle/xte are set to the minimum/maximum x values of the mesh and yle/yte are set to the minimum/maximum y values of the mesh
    2) ble = -bte : sweep without taper not implemented
    """
    if mesh.ndim != 3:
        logger.critical('The mesh to modified must be 3D')
        return -1
    
    nel=mesh.nel
    
    if all(v=='' for v in coord):
        #print('Leading/trailing edge is set to the minimum/maximum x,y-coordinates in the mesh')
        xvec = np.zeros((4*nel,1))
        yvec = np.zeros((4*nel,1))
        it = 0
        for iel in range(nel):
                for i in range(2):
                    for j in range(2):
                        xvec[it] = mesh.elem[iel].pos[0,0,j,i]
                        yvec[it] = mesh.elem[iel].pos[1,0,j,i]
                        it += 1
        indxmin = np.argmin(xvec)
        indxmax = np.argmax(xvec)
        indymin = np.argmin(yvec)
        indymax = np.argmax(yvec)
        xle = xvec[indxmin]
        yle = yvec[indymin]
        xte = xvec[indxmax]
        yte = yvec[indymax]
        zle = z0
        zte = z0
    elif not any(v=='' for v in coord):
        #print('Leading/trailing edge is set to the x,y-coordinates from the user`s definition')
        xle = coord[0]
        yle = coord[1]
        xte = coord[2]
        yte = coord[3]
        zle = z0
        zte = z0
    else:
        logger.critical('The x and y positions of the leading and trailing edges are inconsistent')
        return -9
    
    ble = np.deg2rad(ble)
    bte = np.deg2rad(bte)
    dih = np.deg2rad(dih)
    x1 = np.zeros((2,2,2))
    y1 = np.zeros((2,2,2))
    
    if np.tan(ble)+np.tan(bte) == 0.0:
        if (ble==0.0 and bte==0.0):
            print('No change to mesh inside taper')
            return mesh
        else:
            logger.critical('Sweep without taper not implemented')
            return -2
    else:
        #coordinates of apex of auxiliar triangle
        xa = xle+np.tan(ble)*(xte-xle+np.tan(bte)*zte-np.tan(bte)*zle)/(np.tan(ble)+np.tan(bte))
        za = (xte-xle+np.tan(bte)*zte+np.tan(ble)*zle)/(np.tan(ble)+np.tan(bte))
        ya = 0.5*(yle+yte)+np.tan(dih)*(za-z0)
    
    for (iel, el) in enumerate(mesh.elem):
        # scale and translate x and y coordinates if nonorthogonal extrusion
        x1 = xa+(xa-el.pos[0, :, :, :])/(za-z0)*(el.pos[2, :, :, :]-za)
        y1 = ya+(ya-el.pos[1, :, :, :])/(za-z0)*(el.pos[2, :, :, :]-za)
        el.pos[0, :, :, :] = x1
        el.pos[1, :, :, :] = y1
        for icurv in range(12):
            curv_type = el.ccurv[icurv]
            if curv_type == 'm':
                # scale and translate x and y coordinates if nonorthogonal extrusion
                el.curv[icurv][0] = xa+(xa-el.curv[icurv][0])/(za-z0)*(el.curv[icurv][2]-za)
                el.curv[icurv][1] = ya+(ya-el.curv[icurv][1])/(za-z0)*(el.curv[icurv][2]-za)
            elif curv_type == 'C':
                #find z-position of edge, to rescale radius
                if icurv < 8:
                    [pos1,pos2] = edge_pos(el, icurv)
                    zr = 0.5*(pos1[2]+pos2[2])
                else:
                    print('Warning: Curvature C is not defined for edges 9-12. Nek5000 will ignore it. Ignoring in taper.')
                # scaling the radius of the circle
                el.curv[icurv][0] = el.curv[icurv][0]*abs((zr-za)/(za-z0))
            elif curv_type != '':
                print('Warning: Taper only implemented only for curvatures m and C. Curvature s acts on faces, not consistent with taper.')
    
    # return the modified mesh
    return mesh



#==============================================================================
def edge_mid(el, iedge):
    """Finds the coordinates of the midsize-node of edge iedge of element el (in other words, if the curvature were type 'm', the values of el.curv[iedge][:3]):
    
    Parameters
    ----------
    el : exadata
         element of mesh (usually, el=mesh.elem[i])
    iedge : int
            index of edge
    """
    
    #correct if ccurv=='m', otherwise, works as allocation
    midpoint=copy.deepcopy(el.curv[iedge][:3])
    
    if el.ccurv[iedge] != 'm':
        [pos1,pos2] = edge_pos(el, iedge)
        
        if el.ccurv[iedge] == '':
            midpoint = (pos1+pos2)/2.0
        elif el.ccurv[iedge] == 'C':
            # Curvature 'C' only needs x and y. Works for 2d and extruded meshes.
            if iedge > 7:
                # For iedge=8-11: will give a different value to what Nek considers (Nek ignores it).
                print('Calculating midpoint differently from Nek5000. Nek ignores it for edges 9-12.')
            radius = el.curv[iedge][0]
            dmid = abs(radius)-(radius**2-(pos2[0]-pos1[0])**2/4.0-(pos2[1]-pos1[1])**2/4.0)**0.5
            midpoint[0] = (pos2[0]+pos1[0])/2.0+dmid/((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)**(0.5)*radius/abs(radius)*(pos2[1]-pos1[1])
            midpoint[1] = (pos2[1]+pos1[1])/2.0-dmid/((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)**(0.5)*radius/abs(radius)*(pos2[0]-pos1[0])
            midpoint[2] = (pos2[2]+pos1[2])/2.0
        elif el.ccurv[iedge] == 's':
            # It doesn't check if sphere is consistent with pos1 and pos2. Just assumes it is.
            radius = el.curv[iedge][4]
            center = el.curv[iedge][:3]
            dist = (pos2+pos1)/2.0-center
            midpoint = center + dist*radius/(dist[0]**2+dist[1]**2+dist[2]**2)**0.5
    
    # return the coordinate of midsize node
    return midpoint



#==============================================================================
def edge_circle(el, iedge, midpoint):
    """Finds the radius of curvature and circle center based on the midsize-node of edge iedge of element el:
    
    Parameters
    ----------
    el : exadata
         element of mesh (usually, el=mesh.elem[i])
    iedge : int
            index of edge
    midpoint : float
               list of coordinates of midsize-node (in other words, if the curvature were type 'm', the values of el.curv[iedge][:3])
    """
    
    [pos1,pos2] = edge_pos(el, iedge)
    
    side1 = midpoint-pos1
    side2 = pos2-midpoint
    side3 = pos1-pos2
    
    d1 = (side1[0]**2+side1[1]**2+side1[2]**2)**0.5
    d2 = (side2[0]**2+side2[1]**2+side2[2]**2)**0.5
    d3 = (side3[0]**2+side3[1]**2+side3[2]**2)**0.5
    sper = (d1+d2+d3)/2.0
    area = (sper*(sper-d1)*(sper-d2)*(sper-d3))**0.5
    
    if area > 0.0001*d1*d2:
        radius = d1*d2*d3/(4*area)
        alpha1 = d2**2*(d1**2+d3**2-d2**2)/2.0
        alpha2 = d3**2*(d2**2+d1**2-d3**2)/2.0
        alpha3 = d1**2*(d3**2+d2**2-d1**2)/2.0
        center = (alpha1*pos1+alpha2*midpoint+alpha3*pos2)/(8.0*area**2)
        if ((side1[0]-side3[0])*(side2[1]-side1[1])-(side1[0]-side2[0])*(side3[1]-side1[1])) < 0.0:
            # if curvature == 'C', the radius is negative for clockwise triangles 
            # works only for 2d/extruded mesh - do not know how to interpret it in 3d (should work for edges edges 0-7 of extruded meshes, unknown behaviour for edges 8-11)
            radius = -radius
    else:
        #radius is too big compared to edge. For pratical purposes, no curvature: radius=0.0
        radius = 0.0
        center = [0.0, 0.0, 0.0]
    
    curv = copy.deepcopy(el.curv[iedge][:4])
    curv[0] = radius
    curv[1:4] = center
    
    # return radius and center of curvature (it is not ready to be used: For 'C', define curv[1:4]=0.0; For 's', define el.curv[4]=abs(curv[0]) and el.curv[:3]=curv[1:4])
    return curv



#==============================================================================
def edge_pos(el, iedge):
    """Finds the positions of nodes of edge iedge of element el:
    
    Parameters
    ----------
    el : exadata
         element of mesh (usually, el=mesh.elem[i])
    iedge : int
            index of edge
    """
    
    if iedge == 0:
        pos1 = el.pos[:, 0, 0, 0]
        pos2 = el.pos[:, 0, 0, 1]
    elif iedge == 1:
        pos1 = el.pos[:, 0, 0, 1]
        pos2 = el.pos[:, 0, 1, 1]
    elif iedge == 2:
        pos1 = el.pos[:, 0, 1, 1]
        pos2 = el.pos[:, 0, 1, 0]
    elif iedge == 3:
        pos1 = el.pos[:, 0, 1, 0]
        pos2 = el.pos[:, 0, 0, 0]
    elif iedge == 4:
        pos1 = el.pos[:, 1, 0, 0]
        pos2 = el.pos[:, 1, 0, 1]
    elif iedge == 5:
        pos1 = el.pos[:, 1, 0, 1]
        pos2 = el.pos[:, 1, 1, 1]
    elif iedge == 6:
        pos1 = el.pos[:, 1, 1, 1]
        pos2 = el.pos[:, 1, 1, 0]
    elif iedge == 7:
        pos1 = el.pos[:, 1, 1, 0]
        pos2 = el.pos[:, 1, 0, 0]
    elif iedge == 8:
        pos1 = el.pos[:, 0, 0, 0]
        pos2 = el.pos[:, 1, 0, 0]
    elif iedge == 9:
        pos1 = el.pos[:, 0, 0, 1]
        pos2 = el.pos[:, 1, 0, 1]
    elif iedge == 10:
        pos1 = el.pos[:, 0, 1, 1]
        pos2 = el.pos[:, 1, 1, 1]
    elif iedge == 11:
        pos1 = el.pos[:, 0, 1, 0]
        pos2 = el.pos[:, 1, 1, 0]
    
    # return position of nodes
    return [pos1, pos2]



#==============================================================================
def lim_polyg(mesh, iel0, iedge0):
    """
    Function to define line that intersect all neighboring elements starting from a element iel and edge iedge. It defines a polygon using the midpoints of the intersected edges.
    Keep in mind: if the elements do not form a closed polygon, the user needs to add points to array xyzline (x,y,z of vertices of polygon) by hand to close the polygon.
    Used to define array that is input for fun_polyg.
    ----------
    """
    xyzline = np.empty((0,3))
    xyzline2 = np.empty((0,3))
    
    #iel0 = iel0-1 #transforming Nek index to pymech index
    #iedge0 = iedge0-1 #transforming Nek index to pymech index
    
    iel = iel0+1
    iedge = iedge0
    
    el = mesh.elem[iel0]
    
    iedge = iedge0
    while (el.bcs[0, iedge][0] == 'E' and iel !=iel0):
        xyzline = np.append(xyzline, [edge_mid(el, iedge)], axis=0)
        iel = int(el.bcs[0, iedge][3]-1)
        iedge = int(el.bcs[0, iedge][4]-1)
        el = mesh.elem[iel]
        iedge = (iedge+2)%4
    xyzline = np.append(xyzline, [edge_mid(el, iedge)], axis=0)
    
    if iel !=iel0:
        el = mesh.elem[iel0]
        iedge = iedge0
        iedge = (iedge0+2)%4
        while (el.bcs[0, iedge][0] == 'E' and iel !=iel0):
            xyzline2 = np.append(xyzline2, [edge_mid(el, iedge)], axis=0)
            iel = int(el.bcs[0, iedge][3]-1)
            iedge = int(el.bcs[0, iedge][4]-1)
            el = mesh.elem[iel]
            iedge = (iedge+2)%4
        xyzline2 = np.append(xyzline2, [edge_mid(el, iedge)], axis=0)
        xyzline = np.append(np.flip(xyzline, axis=0), xyzline2, axis=0)
    
    return xyzline



#==============================================================================
def iel_point(mesh, xyzpoint):
    """
    Function to find which element contains point defined in xypoint (does not consider curvature, treats elements as quadrilateral)
    ----------
    """
    #Consistency checks: Initial grid
    if mesh.ndim != 2:
        print('Warning: mesh must be 2D to find which element contains the point.')
    
    ifound = 0
    area = np.zeros((4))
    for (iel, el) in enumerate(mesh.elem):
        for iedge in range(4):
            [pos1,pos2] = edge_pos(el, iedge)
                
            side1 = xyzpoint-pos1
            side2 = pos2-xyzpoint
            side3 = pos1-pos2
            #Calculating signed area of triangle
            area[iedge] = 0.5*((side1[0]-side3[0])*(side2[1]-side1[1])-(side1[0]-side2[0])*(side3[1]-side1[1]))
        #If both areas with oposing edges have the same sign, it is in the inside the element
        if (area[0]*area[2] > 0.0 and area[1]*area[3] > 0.0):
            ielpoint = iel
            if ifound > 0:
                print('Warning: point is inside more than one element.')
            ifound = ifound+1
    
    return ielpoint



#==============================================================================
def iface_neig(mesh, iel, ielneig):
    """
    Function to find which face of mesh.elem[iel] is connected to mesh.elem[ielneig]
    ----------
    """
    
    el = mesh.elem[iel]
    ifaceneig = ''
    
    for iface in range(6):
        if (el.bcs[0, iface][0] == 'E') and (ielneig == int(el.bcs[0, iface][3]-1)):
            ifaceneig = int(el.bcs[0, iface][4]-1)
    
    if ifaceneig == '':
        logger.critical('Neighboring face was not found. Elements are not neighbours or bc not equal to E')
        return ''
    
    return ifaceneig



#==============================================================================
def fun_polyg(xpos, ypos, xyzline):
    """
    Function to define high discretization mesh and low discretization mesh.
    Given x and y-position (xpos, ypos) of a point of a 2d mesh, funval can be positive or negative
    Internal mesh: funval should be negative. External mesh: funval should be positive (the definition of internal/external is interchangeable, but they must be consistent if multiple functions are defined).
    The boundary defined by this function should cross the mid elements such that exactly 2 points are in each side of the boundary (2 points inside and 2 points are outside).
    For fun_polyg, the boundary is a polygon of any form formed by the positions xyzline (xyzline may be x,y,z or just x,y).
    ----------
    """
    
    xyzline = xyzline[:,:2]
    xypos= np.zeros((1,2))
    xypos[0,0] = xpos
    xypos[0,1] = ypos
    
    path = mpltPath.Path(xyzline)
    inside = path.contains_points(xypos)
    
    if inside:
        funval = -1.0
    else:
        funval = 1.0
    
    return funval



#==============================================================================
def define_z(z, n, extyp = 'uni', verbose = 2):
    """Define z positions (zlist) for extrusion functions:
    
    Parameters
    ----------
    z : float
        list [zmin, zmax, dz0, r] with values of minimum and maximum z, the initial grid distance and the ratio of the geometric progression
    n : int
        number of elements per interval of the extruded mesh. Number of nodes is n+1
    extyp : string
            extrusion type in z (uniform, geometric progression, read file, etc)
    verbose : int
              level of verbosity. 0: only error and warning messages; 1: important messages; 2: all.
    
    Special cases
    -------------
    1) extyp == 'uni'  : uniform, given zmin, zmax and n
    2) extyp == 'unidz': uniform, given zmin, dz and n
    3) extyp == 'gprn' : geometric progression, given zmin, zmax, ratio and n
    4) extyp == 'gpdzr': geometric progression, given zmin, zmax, dz0 and ratio
    5) extyp == 'gpdzn': geometric progression, given zmin, zmax, dz0 and n
    6) extyp == 'txt'  : read file z.txt with z positions
    7) extyp == 'txt1' : given zmin and zmax, read file z.txt with normalized z positions (this option can use to scale and translate the values in txt by any values, even if z.txt is not normalized)
    """
    
    #Defining z positions for extrusion
    # uniform, given zmin, zmax and n
    if extyp == 'uni':
        if verbose > 1:
            print('Uniform extrusion. zmin=', z[0], '; zmax=', z[1], '; nelz=', n, '; dz=', (z[1]-z[0])*1.0/n)
        zlist = z[0]+(z[1]-z[0])*np.arange(n+1)/n
    
    # uniform, given zmin, dz and n
    elif extyp == 'unidz':
        if verbose > 1:
            print('Uniform extrusion. zmin=', z[0], '; zmax=', z[0] + z[2]*n, '; nelz=', n, '; dz=', z[2])
        zlist = z[0] + z[2]*np.arange(n+1)
    
    # geometric progression, given zmin, zmax, ratio and n
    elif extyp == 'gprn':
        rgp = z[3]
        if rgp != 1.0:
            dz0 = (z[1]-z[0])*(1.0-rgp)/(1.0-rgp**n)
            if verbose > 1:
                print('Geometric progression extrusion. zmin=', z[0], '; zmax=', z[1], '; nelz=', n, '; dz0=', dz0, '; r=', rgp)
            zlist = z[0] + dz0*(1.0-rgp**np.arange(n+1))/(1.0-rgp)
        else:
            if verbose > 1:
                print('Uniform extrusion. zmin=', z[0], '; zmax=', z[1], '; nelz=', n, '; dz=', (z[1]-z[0])*1.0/n)
            zlist = z[0]+(z[1]-z[0])*np.arange(n+1)/n
    
    # geometric progression, given zmin, zmax, dz0 and ratio
    elif extyp == 'gpdzr':
        rgp = z[3]
        if rgp != 1.0:
            dz0 = z[2]
            n = round(np.log(1.0 - (z[1]-z[0])*(1.0-rgp)/dz0)/np.log(rgp))
            if verbose > 1:
                print('Geometric progression extrusion. zmin=', z[0], '; zmax=', z[1], '; nelz=', n, '; dz0=', dz0, '; r=', rgp)
            zlist = z[0] + dz0*(1.0-rgp**np.arange(n+1))/(1.0-rgp)
            if zlist[-1]!=z[1]:
                if verbose > 0:
                    print('Extrusion exceeds or does not reach requested zmax. Reducing or increasing', zlist[-1],' to', z[1])
                zlist[-1] = z[1]
        else:
            if verbose > 1:
                print('Uniform extrusion. zmin=', z[0], '; zmax=', z[0] + z[1]*n, '; nelz=', n, '; dz=', z[1])
            zlist = z[0] + z[2]*np.arange(n+1)
    
    # geometric progression, given zmin, zmax, dz0 and n
    elif extyp == 'gpdzn':
        dz0 = z[2]
        f_rgp = lambda r: (z[1]-z[0])-(dz0*(1.0-r**n)/(1.0-r))
        nuni = (z[1]-z[0])/dz0
        if nuni <= 0.0:
            logger.critical('dz does not have the same sign of zmax-zmin')
            return -4
        elif abs(nuni - n) < 1e-6:
            if verbose > 1:
                print('Extrusion is pratically uniform. Changing to uniform extrusion. zmin=', z[0], '; zmax=', z[1], '; nelz=', n, '; dz=', (z[1]-z[0])*1.0/n)
            zlist = z[0]+(z[1]-z[0])*np.arange(n+1)/n
        else:
            if nuni < n:
                rgp = sp.optimize.root_scalar(f_rgp, bracket=[0.001, 1.0-1e-10], x0=0.8, x1=0.999).root
            elif nuni > n:
                rgp = sp.optimize.root_scalar(f_rgp, bracket=[1.0+1e-10, 1000.0], x0=1.2, x1=1.001).root
            if verbose > 1:
                print('Geometric progression extrusion. zmin=', z[0], '; zmax=', z[1], '; nelz=', n, '; dz0=', dz0, '; r=', rgp)
            zlist = z[0] + dz0*(1.0-rgp**np.arange(n+1))/(1.0-rgp)
            if zlist[-1]!=z[1]:
                if verbose > 0:
                    print('Extrusion exceeds or does not reach requested zmax. Reducing or increasing', zlist[-1],' to', z[1])
                zlist[-1] = z[1]
    
    # read file z.txt with z positions
    elif extyp == 'txt':
        if verbose > 1:
            print('Reading z-point distribution from z.txt file')
        f = open('z.txt')
        lines = f.readlines()
        f.close
        nz = len(lines)
        zlist = np.zeros(nz)
        for k in range(0,nz):
            zlist[k] = float(lines[k])
        if len(zlist) < 1:
            logger.critical('At least two points are necessary in the z.txt file')
            return -5
    
    # given zmin and zmax, read file z.txt with normalized z positions
    elif extyp == 'txt1':
        if verbose > 1:
            print('Reading z-point distribution (scaled from 0 to 1) from z.txt file')
        f = open('z.txt')
        lines = f.readlines()
        f.close
        nz = len(lines)
        zlist = np.zeros(nz)
        for k in range(0,nz):
            zlist[k] = z[0]+float(lines[k])*(z[1]-z[0])
        if len(zlist) < 1:
            logger.critical('At least two points are necessary in the z.txt file')
            return -5
    
    # return the z positions
    return zlist



#=================================================================================
def update_ncurv(mesh):
    """Counts the number of curved sides (edges or faces)
    
    Parameters
    ----------
    mesh : exadata
           The mesh to count.
    """

    ncurv = 0
    for el in mesh.elem:
        for iedge in range(12):
            if el.ccurv[iedge] != '':
                ncurv = ncurv+1
    mesh.ncurv = ncurv
    
    return ncurv



#==============================================================================
def merge(mesh, other, tol=1e-9, ignore_empty=True, ignore_connect=True):
    """
    Merges mesh to other and store it in mesh_merg, and connects it
    
    Parameters
    ----------
    other: exadata
        mesh to merge into mesh
    
    tol: float
        maximum distance at which points are considered identical
    
    ignore_empty: bool
        if True, the faces with an empty boundary condition ('') will be treated as internal faces and will not be merged.
        This is useful if internal boundary conditions are not defined and will make the operation *much* faster,
        but requires boundary conditions to be defined on the faces to be merged.
    
    ignore_connect: bool
        if True, the code will not try to resolve conflits or missing internal or periodic boundary conditions
    """
    
    # perform some consistency checks
    if mesh.ndim != other.ndim:
        logger.error(f'Cannot merge meshes of dimensions {mesh.ndim} and {other.ndim}!')
        return -1
    if mesh.lr1[0] != other.lr1[0]:
        logger.error('Cannot merge meshes of different polynomial orders ({} != {})'.format(mesh.lr1[0], other.lr1[0]))
        return -2
    
    # add the new elements (in an inconsistent state if there are internal boundary conditions)
    mesh_merg = copy.deepcopy(mesh)
    nel1 = mesh.nel
    mesh_merg.nel = mesh.nel + other.nel
    mesh_merg.ncurv = mesh.ncurv + other.ncurv
    mesh_merg.elmap = np.concatenate((mesh.elmap, other.elmap+nel1))
    # the deep copy is required here to avoid leaving the 'other' mesh in an inconsistent state by modifying its elements
    mesh_merg.elem = mesh_merg.elem + copy.deepcopy(other.elem)
    
    # check how many boundary condition fields we have
    nbc = min(mesh.nbc, other.nbc)
    
    # correct the boundary condition numbers:
    # the index of the neighbours have changed
    for iel in range(nel1, mesh_merg.nel):
        for ibc in range(other.nbc):
            for iface in range(6):
                bc = mesh_merg.elem[iel].bcs[ibc, iface][0]
                if bc == 'E' or bc == 'P':
                    neighbour = mesh_merg.elem[iel].bcs[ibc, iface][3]
                    mesh_merg.elem[iel].bcs[ibc, iface][3] = neighbour + nel1
    
    # glue common faces together
    # only look for the neighbour in the first BC field because it should be the same in all fields.
    # FIXME: this will fail to correct periodic conditions if periodic domains are merged together.
    nchanges = 0  # counter for the boundary conditions connected
    if not ignore_connect:
        nfaces = 2*mesh_merg.ndim
        if nbc > 0:
            for iel in range(nel1, mesh_merg.nel):
                for iface in range(nfaces):
                    bc = mesh_merg.elem[iel].bcs[0, iface][0]
                    if bc != 'E' and not (ignore_empty and bc == ''):
                        # boundary element, look if it can be connected to something
                        for iel1 in range(nel1):
                            for iface1 in range(nfaces):
                                bc1 = mesh_merg.elem[iel1].bcs[0, iface1][0]
                                if bc1 != 'E' and not (ignore_empty and bc1 == ''):
                                    # if the centers of the faces are close, connect them together
                                    x0, y0, z0 =  mesh_merg.elem[iel ].face_center(iface)
                                    x1, y1, z1 =  mesh_merg.elem[iel1].face_center(iface1)
                                    dist2 = (x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2
                                    if dist2 <= tol**2:
                                        # reconnect the periodic faces together (assumes that all fields are periodic)
                                        if bc == 'P' and bc1 == 'P':
                                            iel_p    = int(mesh_merg.elem[iel ].bcs[0, iface ][3])-1
                                            iel_p1   = int(mesh_merg.elem[iel1].bcs[0, iface1][3])-1
                                            iface_p  = int(mesh_merg.elem[iel ].bcs[0, iface ][4])-1
                                            iface_p1 = int(mesh_merg.elem[iel1].bcs[0, iface1][4])-1
                                            for ibc in range(nbc):
                                                mesh_merg.elem[iel_p ].bcs[ibc, iface_p ][0] = 'P'
                                                mesh_merg.elem[iel_p1].bcs[ibc, iface_p1][0] = 'P'
                                                mesh_merg.elem[iel_p ].bcs[ibc, iface_p ][3] = iel_p1+1
                                                mesh_merg.elem[iel_p1].bcs[ibc, iface_p1][3] = iel_p +1
                                                mesh_merg.elem[iel_p ].bcs[ibc, iface_p ][4] = iface_p1+1
                                                mesh_merg.elem[iel_p1].bcs[ibc, iface_p1][4] = iface_p +1
                                        for ibc in range(nbc):
                                            mesh_merg.elem[iel ].bcs[ibc, iface ][0] = 'E'
                                            mesh_merg.elem[iel1].bcs[ibc, iface1][0] = 'E'
                                            mesh_merg.elem[iel ].bcs[ibc, iface ][3] = iel1+1
                                            mesh_merg.elem[iel1].bcs[ibc, iface1][3] = iel +1
                                            mesh_merg.elem[iel ].bcs[ibc, iface ][4] = iface1+1
                                            mesh_merg.elem[iel1].bcs[ibc, iface1][4] = iface +1
                                        nchanges = nchanges+1
        logger.debug(f'merged {nchanges} pairs of faces')
    return mesh_merg



#=================================================================================
def delete_internal_bcs(mesh):
    """Deletes the internal boundary conditions 'E' in a Nek5000 mesh.
    Those are present in .rea files but do not need to be valid, and are completely absent from .re2 files.
    Returns the number of deleted conditions.
    
    Parameters
    ----------
    mesh : exadata
           The mesh to modify in-place.
    """

    ndelete = 0
    for ibc in range(mesh.nbc):
        for (iel, el) in enumerate(mesh.elem):
            for iface in range(6):
                bc = el.bcs[ibc, iface]
                if bc[0] == 'E':
                    ndelete = ndelete+1
                    bc[0] = ''
                    bc[1] = iel+1
                    bc[2] = iface+1
                    for i in range(3, 8):
                        bc[i] = 0
    return ndelete



#=================================================================================
def generate_internal_bcs(mesh, tol=1e-3):
    """Generates internal boundary conditions 'E' in a Nek5000 mesh based on geometric proximity of faces.
    This will loop over all pairs of faces and connect them if their centres are closer than some specified relative tolerance.
    This function returns the number of internal connections found.

    Parameters
    ----------
    mesh : exadata
           The mesh to modify in-place
    
    tol: float
         the tolerance within which the centres of two faces are considered the same,
         relative to the smallest edge of the elements
    """

    # First generate a length scale (squared) for each element, equal to the square of the smallest edge of tha element.
    scales = np.zeros((mesh.nel,))
    for (iel, (el, l2)) in enumerate(zip(mesh.elem, scales)):
        # get coordinates of points
        x1, y1, z1 = el.pos[:, 0, 0, 0]
        x2, y2, z2 = el.pos[:, 0, 0, 1]
        x3, y3, z3 = el.pos[:, 0, 1, 0]
        x4, y4, z4 = el.pos[:, 0, 1, 1]
        # compute squares of edges lengths
        edges_l2 = np.zeros((12,))
        edges_l2[0] = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
        edges_l2[1] = (x3-x2)**2 + (y3-y2)**2 + (z3-z2)**2
        edges_l2[2] = (x4-x3)**2 + (y4-y3)**2 + (z4-z3)**2
        edges_l2[3] = (x1-x4)**2 + (y1-y4)**2 + (z1-z4)**2
        if mesh.ndim > 2:
            # in 3D, do the same for the upper face, and also the side edges
            x5, y5, z5 = el.pos[:, 1, 0, 0]
            x6, y6, z6 = el.pos[:, 1, 0, 1]
            x7, y7, z7 = el.pos[:, 1, 1, 0]
            x8, y8, z8 = el.pos[:, 1, 1, 1]
            edges_l2[4] = (x6-x5)**2 + (y6-y5)**2 + (z6-z5)**2
            edges_l2[5] = (x7-x6)**2 + (y7-y6)**2 + (z7-z6)**2
            edges_l2[6] = (x8-x7)**2 + (y8-y7)**2 + (z8-z7)**2
            edges_l2[7] = (x5-x8)**2 + (y5-y8)**2 + (z5-z8)**2
            edges_l2[8] = (x5-x1)**2 + (y5-y1)**2 + (z5-z1)**2
            edges_l2[9] = (x6-x2)**2 + (y6-y2)**2 + (z6-z2)**2
            edges_l2[10] = (x7-x3)**2 + (y7-y3)**2 + (z7-z3)**2
            edges_l2[11] = (x8-x4)**2 + (y8-y4)**2 + (z8-z4)**2
            l2 = edges_l2.min()
        else:
            l2 = edges_l2[:4].min()

        # check if there is a zero length edge; in this case the mesh is invalid and there is no point continuing.
        if l2 <= 0.0:
            logger.critical(f'Detected an edge with zero length in element {iel}!')
            return -1
    
    # Now that we have the scales, we can compare the location of the faces for each pair of elements and connect them if they are close
    nconnect = 0  # number of connections made
    for iel in range(mesh.nel):
        el = mesh.elem[iel]
        l2 = scales[iel]
        for other_iel in range(iel+1, mesh.nel):
            other_el = mesh.elem[other_iel]
            other_l2 = scales[other_iel]
            max_d2 = tol**2*min(l2, other_l2)
            for iface in range(2*mesh.ndim):
                xf, yf, zf = el.face_center(iface)
                for other_iface in range(2*mesh.ndim):
                    other_xf, other_yf, other_zf = other_el.face_center(other_iface)
                    dist2 = (other_xf-xf)**2 + (other_yf-yf)**2 + (other_zf-zf)**2 
                    if dist2 <= max_d2:
                        for ibc in range(mesh.nbc):
                            # increment counter for diagnostics
                            nconnect = nconnect+1
                            # write the connectivity information in both directions
                            el.bcs[ibc, iface][0] = 'E'
                            el.bcs[ibc, iface][1] = iel+1
                            el.bcs[ibc, iface][2] = iface+1
                            el.bcs[ibc, iface][3] = other_iel+1
                            el.bcs[ibc, iface][4] = other_iface+1
                            other_el.bcs[ibc, other_iface][0] = 'E'
                            other_el.bcs[ibc, other_iface][1] = other_iel+1
                            other_el.bcs[ibc, other_iface][2] = other_iface+1
                            other_el.bcs[ibc, other_iface][3] = iel+1
                            other_el.bcs[ibc, other_iface][4] = iface+1
    
    return nconnect
